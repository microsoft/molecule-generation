"""Decoding layer for a CGVAE."""
import heapq
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import tensorflow as tf
from dpu_utils.tf2utils import MLP, unsorted_segment_log_softmax
from tf2_gnn.layers import (
    GNN,
    GNNInput,
    NodesToGraphRepresentationInput,
    WeightedSumGraphRepresentation,
)

from molecule_generation.utils.beam_utils import (
    Ray,
    RayExtension,
    ExtensionType,
    MoleculeGenerationEdgeCandidateInfo,
    MoleculeGenerationStepInfo,
    extend_beam,
)
from molecule_generation.chem.topology_features import calculate_topology_features
from molecule_generation.chem.valence_constraints import (
    constrain_edge_choices_based_on_valence,
    constrain_edge_types_based_on_valence,
)
from molecule_generation.chem.atom_feature_utils import AtomFeatureExtractor
from molecule_generation.preprocessing.cgvae_generation_trace import (
    NodeState,
    calculate_dist_from_focus_to_valid_target,
)

_COMPAT_DISTANCE_TRUNCATION = 100


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    ]
)
def traced_unsorted_segment_log_softmax(
    logits: tf.Tensor, segment_ids: tf.Tensor, num_segments: tf.Tensor
):
    return unsorted_segment_log_softmax(logits, segment_ids, num_segments)


class CGVAEDecoderInput(NamedTuple):
    """Named tuple for CGVAEDecoderCell"""

    node_features: tf.Tensor
    adjacency_lists: Tuple[tf.Tensor, ...]
    num_partial_graphs_in_batch: tf.Tensor
    graph_to_focus_node_map: tf.Tensor
    node_to_graph_map: tf.Tensor
    valid_edge_choices: tf.Tensor
    edge_features: tf.Tensor


class CGVAEDecoder(tf.keras.layers.Layer):
    """Decode graph states using a combination of graph message passing layers and dense layers

    Example usage:
    >>> layer_input = CGVAEDecoderInput(
    ...     node_features=tf.random.normal(shape=(5, 12)),
    ...     adjacency_lists=(
    ...         tf.constant([[0, 1], [1, 2], [3, 4]], dtype=tf.int32),
    ...         tf.constant([[1, 2], [3, 4]], dtype=tf.int32),
    ...         tf.constant([[2, 0]], dtype=tf.int32),
    ...     ),
    ...     num_partial_graphs_in_batch=tf.constant(2, dtype=tf.int32),
    ...     graph_to_focus_node_map=tf.constant([0, 3], dtype=tf.int32),
    ...     node_to_graph_map=tf.constant([0, 0, 0, 1, 1], dtype=tf.int32),
    ...     valid_edge_choices=tf.constant([[0, 1], [0, 2], [3, 4]]),
    ...     edge_features=tf.constant([[0.6], [0.1], [0.9]])
    ... )
    >>> params = CGVAEDecoder.get_default_params()
    >>> from molecule_generation.chem.atom_feature_utils import AtomTypeFeatureExtractor
    >>> atom_type_featuriser = AtomTypeFeatureExtractor()
    >>> layer = CGVAEDecoder(params, use_self_loop_edges_in_partial_graphs=True, feature_extractors=[atom_type_featuriser])
    >>> output = layer(layer_input)
    >>> print(output)
    (<tf.Tensor... shape=(5, 1), dtype=float32, ...>, <tf.Tensor... shape=(3, 3), dtype=float32, ...>)
    """

    @classmethod
    def get_default_params(cls, mp_style: Optional[str] = None) -> Dict[str, Any]:
        gnn_params = GNN.get_default_hyperparameters(mp_style)
        decoder_gnn_params = {"gnn_" + k: v for k, v in gnn_params.items()}
        these_params = {
            "gnn_hidden_dim": 64,
            "gnn_num_layers": 12,
            "gnn_message_activation_function": "leaky_relu",
            "gnn_residual_every_num_layers": 10000,  # Turn off, as we are reading out all layers anyway
            "gnn_global_exchange_every_num_layers": 10000,  # Turn off, as global properties are not useful in this task
            "gnn_dense_every_num_layers": 4,
            "gnn_dense_intermediate_layer_activation": "relu",  # matches default message activation function
            "gnn_layer_input_dropout_rate": 0.0,
            "gnn_global_exchange_dropout_rate": 0.0,
            "global_repr_size": 128,
            "global_repr_weighting_fun": "softmax",  # One of "softmax", "sigmoid"
            "global_repr_num_heads": 4,
            "global_repr_dropout_rate": 0.0,
            "num_edge_types_to_classify": 3,
            "edge_candidate_scorer_hidden_layers": [64, 32],
            "edge_type_selector_hidden_layers": [32],
            "edge_selection_dropout_rate": 0.0,
            "distance_truncation": 10,
            # When training a new model, this should probably be false.
            "compatible_distance_embedding_init": True,
        }
        decoder_gnn_params.update(these_params)
        return decoder_gnn_params

    def __init__(
        self,
        params,
        use_self_loop_edges_in_partial_graphs: bool,
        feature_extractors: List[AtomFeatureExtractor],
        node_type_loss_weights: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """Initialise the layer."""
        super().__init__(**kwargs)
        self._params = params
        self._use_self_loop_edges_in_partial_graphs = use_self_loop_edges_in_partial_graphs
        self._num_edge_types_to_classify = params["num_edge_types_to_classify"]

        self._feature_extractors = feature_extractors
        self._node_type_loss_weights = node_type_loss_weights

        # The number of node types is the number of atom types we saw during training:
        atom_type_featuriser = next(
            featuriser for featuriser in feature_extractors if featuriser.name == "AtomType"
        )
        self._num_node_types = atom_type_featuriser.feature_width

        # Layers to be built.
        gnn_params = {k[4:]: v for k, v in params.items() if k.startswith("gnn_")}
        self._gnn = GNN(gnn_params)

        self._graph_representation_layer = WeightedSumGraphRepresentation(
            graph_representation_size=self._params["global_repr_size"],
            num_heads=self._params["global_repr_num_heads"],
            weighting_fun=self._params["global_repr_weighting_fun"],
            scoring_mlp_dropout_rate=self._params["global_repr_dropout_rate"],
            transformation_mlp_dropout_rate=self._params["global_repr_dropout_rate"],
        )

        self._edge_candidate_scorer = MLP(
            out_size=1,
            hidden_layers=self._params["edge_candidate_scorer_hidden_layers"],
            dropout_rate=self._params["edge_selection_dropout_rate"],
        )

        self._edge_type_selector = MLP(
            out_size=self.num_edge_types_to_classify,
            hidden_layers=self._params["edge_type_selector_hidden_layers"],
            use_biases=True,
            dropout_rate=self._params["edge_selection_dropout_rate"],
        )

        self._stop_node_representation: tf.Variable = None
        self._distance_embedding = None

    @property
    def num_edge_types_to_classify(self):
        return self._num_edge_types_to_classify

    def build(self, tensor_shapes: CGVAEDecoderInput):
        edge_feature_dim = tensor_shapes.edge_features[-1]
        # We extend the initial node features by an is-focus-node-bit, so prepare that shape here:
        focus_node_bit_size = 1
        initial_node_feature_shape = tf.TensorShape(
            dims=(None, tensor_shapes.node_features[-1] + focus_node_bit_size)
        )

        with tf.name_scope("decoder_gnn"):
            # No need to generalise input shapes here. Taken care of by GNN layer.
            gnn_input_shapes = GNNInput(
                node_features=initial_node_feature_shape,
                adjacency_lists=tensor_shapes.adjacency_lists,
                node_to_graph_map=tensor_shapes.node_to_graph_map,
                num_graphs=tensor_shapes.num_partial_graphs_in_batch,
            )
            self._gnn.build(gnn_input_shapes)

        # We get the initial GNN input (after projection) + results for all layers:
        node_repr_size = self._params["gnn_hidden_dim"] * (1 + self._params["gnn_num_layers"])

        with tf.name_scope("global_repr"):
            final_node_representation_shapes = NodesToGraphRepresentationInput(
                node_embeddings=tf.TensorShape(dims=(None, node_repr_size)),
                node_to_graph_map=tensor_shapes.node_to_graph_map,
                num_graphs=tensor_shapes.num_partial_graphs_in_batch,
            )
            self._graph_representation_layer.build(final_node_representation_shapes)

        # Edge candidates are represented by a graph-global representation,
        # the representations of source and target nodes, and edge features:
        edge_candidate_representation_size = (
            self._params["global_repr_size"] + 2 * node_repr_size + edge_feature_dim
        )
        with tf.name_scope("edge_candidate_scorer"):
            self._edge_candidate_scorer.build(
                input_shape=[None, edge_candidate_representation_size]
            )
            self._stop_node_representation = self.add_weight(
                name="StopNodeRepr",
                shape=(1, node_repr_size + edge_feature_dim),
                trainable=True,
            )

        with tf.name_scope("edge_type_selector"):
            self._edge_type_selector.build(input_shape=[None, edge_candidate_representation_size])

        self._build_distance_embedding_weight()

        super().build(tensor_shapes)

        variable_node_features_shape = tf.TensorShape((None, tensor_shapes.node_features[-1]))
        call_input_spec = (
            CGVAEDecoderInput(
                node_features=tf.TensorSpec(shape=variable_node_features_shape, dtype=tf.float32),
                adjacency_lists=tuple(
                    tf.TensorSpec(shape=(None, 2), dtype=tf.int32)
                    for _ in range(len(tensor_shapes.adjacency_lists))
                ),
                num_partial_graphs_in_batch=tf.TensorSpec(shape=(), dtype=tf.int32),
                graph_to_focus_node_map=tf.TensorSpec(shape=(None,), dtype=tf.int32),
                node_to_graph_map=tf.TensorSpec(shape=(None,), dtype=tf.int32),
                valid_edge_choices=tf.TensorSpec(shape=(None, 2), dtype=tf.int32),
                edge_features=tf.TensorSpec(shape=(None, edge_feature_dim), dtype=tf.float32),
            ),
            tf.TensorSpec(shape=(), dtype=tf.bool),
        )
        setattr(self, "call", tf.function(func=self.call, input_signature=call_input_spec))

    def _build_distance_embedding_weight(self):
        """Utility method to build the distance embedding weights.

        Should be used directly only in tests. Normally called through the build method.
        """
        # Guard against calling twice.
        if self._distance_embedding is not None:
            return

        with tf.name_scope("distance_embedding"):
            # Temporary backwards compatibility code.
            if self._params.get("compatible_distance_embedding_init", True):

                def _initializer(*args, **kwargs):
                    return np.arange(
                        start=0,
                        stop=self._params.get("distance_truncation", _COMPAT_DISTANCE_TRUNCATION),
                        dtype=np.float32,
                    ).reshape(-1, 1)

            else:
                _initializer = None

            self._distance_embedding = self.add_weight(
                name="distance_embedding",
                shape=(self._params.get("distance_truncation", _COMPAT_DISTANCE_TRUNCATION), 1),
                trainable=True,
                initializer=_initializer,
            )

    def call(self, inputs: CGVAEDecoderInput, training: bool = False):
        """

        Args:
            inputs: A CGVAEDecoderInput tuple. The elements have the following meanings:
                node_features: the encoded node features of the nodes in the partial graph.
                    Shape: [PV, ED]
                adjacency_lists: the adjacency lists of the partial graphs. Each has shape: [E, 2]
                num_partial_graphs_in_batch: the number of partial graphs in the batch. Should be
                    equal to PG.
                graph_to_focus_node_map: the index of the focus node for that partial graph.
                    Shape: [PG,]
                node_to_graph_map: the index of the partial graph that this node came from.
                    Shape: [PV,], with values in the range [0, ..., PG - 1]
                partial_node_to_original_node_map: A mapping from the partial node index to the
                    index of the node in the original graph.
                    Shape: [PV,] with values in the range [0, ..., V - 1]
                valid_edge_choices: the edge choices which are allowable at the current step. A
                    tensor of rank 2. The first column is the source node indices, the second is the
                    target node indices. Shape: [VE, 2]
                edge_features: features associated with the valid edges. A tensor of rank 2, where
                    the first dimension matches the number of valid edges, and the second is the
                    number of edges features. Shape: [VE, FD]

            training: A bool denoting whether this is a training call.

        Returns:
            A tuple of three tensors. The first represents the edge choice logits, which has shape
            [VE + PG, 1]. The first VE elements of the array correspond to the logits for the
            focus_node -> valid edge to be added. The final PG elements correspond to the logits
            for the stop node for each of the PG partial graphs (in order).
            The second represents the edge type logits, which has shape [VE, ET]
            The third represents the node classification logits, which has shape [PV, NT]

        Here:
        - PV = number of _P_artial Graph _V_ertices
        - VD = GNN _v_ertex representation _d_imension
        - GD = graph representation _d_imension
        - FD = edge _f_eature _d_imension
        - PG = number of _P_artial _G_raphs
        - VE = number of _V_alid _E_dges
        - NT = num node types
        - ET = number of edge types
        """
        node_is_in_focus_bit = tf.scatter_nd(
            indices=tf.expand_dims(inputs.graph_to_focus_node_map, axis=-1),
            updates=tf.ones(
                shape=(tf.shape(inputs.graph_to_focus_node_map)[0], 1), dtype=tf.float32
            ),
            shape=(tf.shape(inputs.node_features)[0], 1),
        )
        initial_node_features = tf.concat([inputs.node_features, node_is_in_focus_bit], axis=-1)

        # Encode the initial node representations using the partial adjacency lists:
        encoder_input = GNNInput(
            node_features=initial_node_features,
            adjacency_lists=inputs.adjacency_lists,
            node_to_graph_map=inputs.node_to_graph_map,
            num_graphs=inputs.num_partial_graphs_in_batch,
        )
        _, node_representations_at_all_layers = self._gnn(
            encoder_input, training=training, return_all_representations=True
        )

        node_representations = tf.concat(
            node_representations_at_all_layers, axis=-1
        )  # Shape [V, VD*(num_layers+1)]

        # Calculate graph representations:
        graph_representation_layer_input = NodesToGraphRepresentationInput(
            node_embeddings=node_representations,
            node_to_graph_map=inputs.node_to_graph_map,
            num_graphs=inputs.num_partial_graphs_in_batch,
        )
        graph_representations = self._graph_representation_layer(
            graph_representation_layer_input, training=training
        )  # shape: [PG, GD]

        # Gather up focus node representations:
        focus_node_representations = tf.gather(
            node_representations, inputs.graph_to_focus_node_map
        )  # shape: [PG, VD*(num_layers+1)]

        graph_and_focus_node_representations = tf.concat(
            [graph_representations, focus_node_representations], axis=-1
        )  # shape: [PG, GD + VD*(num_layers+1)]

        # We have to do a couple of gathers here ensure that we decode only the valid nodes.
        valid_target_indices = inputs.valid_edge_choices[:, 1]  # shape: [VE]
        valid_target_to_graph_map = tf.gather(
            inputs.node_to_graph_map, valid_target_indices
        )  # shape: [VE]

        graph_and_focus_node_representations_per_edge_candidate = tf.gather(
            graph_and_focus_node_representations, valid_target_to_graph_map
        )  # shape: [VE, GD + VD*(num_layers+1)]

        # Extract the features for the valid edges only.
        edge_candidate_target_node_representations = tf.gather(
            node_representations, valid_target_indices
        )  # shape: [VE, VD*(num_layers+1)]

        # The zeroth element of edge_features is the graph distance. We need to look that up in the embedding weights.
        truncated_distances = tf.minimum(
            tf.cast(inputs.edge_features[:, 0], dtype=np.int32),
            self._params.get("distance_truncation", _COMPAT_DISTANCE_TRUNCATION) - 1,
        )  # shape: [VE]
        distance_embedding = tf.nn.embedding_lookup(
            self._distance_embedding, truncated_distances
        )  # shape: [VE, 1]
        # Concatenate all the node features, to form focus_node -> target_node edge features
        edge_candidate_representation = tf.concat(
            [
                graph_and_focus_node_representations_per_edge_candidate,
                edge_candidate_target_node_representations,
                distance_embedding,
                inputs.edge_features[:, 1:],
            ],
            axis=-1,
        )  # shape: [VE, GD + 2 * VD*(num_layers+1) + FD]

        # Calculate the stop node features as well.
        stop_edge_selection_representation = tf.concat(
            [
                graph_and_focus_node_representations,
                tf.tile(
                    self._stop_node_representation,
                    multiples=(inputs.num_partial_graphs_in_batch, 1),
                ),
            ],
            axis=-1,
        )  # shape: [PG, GD + 2 * VD*(num_layers+1) + FD]

        edge_candidate_and_stop_features = tf.concat(
            [edge_candidate_representation, stop_edge_selection_representation], axis=0
        )  # shape: [VE + PG, GD + 2 * VD*(num_layers+1) + FD]

        edge_candidate_logits = self._edge_candidate_scorer(
            edge_candidate_and_stop_features, training=training
        )  # shape: [VE + PG, 1]
        edge_type_logits = self._edge_type_selector(
            edge_candidate_representation, training=training
        )  # shape: [VE, ET]

        return edge_candidate_logits, edge_type_logits

    def beam_decode(
        self,
        node_types: List[str],
        node_features: np.ndarray,
        beam_size: int,
        adjacency_lists: Optional[List[np.ndarray]] = None,
        connection_nodes: Optional[List[int]] = None,
        symmetrize_adjacency_lists: bool = True,
        store_generation_traces: bool = False,
    ) -> List[Ray]:
        """Beam decoder for a set of nodes.

        Args:
            node_types: a list of string representations of the node types.
            node_features: numpy array or tf tensor containing the encoded node features. It should
                have shape [V, ED], where V is the number of nodes and ED is the encoded dimension.
            beam_size: the size of the beam to use while searching.
            adjacency_lists: a list of numpy arrays corresponding to the adjacency lists with which we want to seed the
                the ray. All ray updates will build on these adjacency lists.
            connection_nodes: a subset of the nodes in the adjacency lists, to which we can connect new atoms.
            symmetrize_adjacency_lists: a bool representing whether we need to symmetrize the adjacency lists. If the
                adjacency lists are symmetric (i.e. for every edge (u, v) in the adjacency lists, the edge (v, u) is
                is also in the adjacency list), this should be false. Otherwise it must be true.
            store_generation_traces: bool flag indicating if all intermediate steps and decisions
                should be recorded; for example for visualisations and debugging purposes.

        Returns:
            A list of length beam_size containing the most likely graphs given the node features.
        """
        # Set up empty beams.
        num_nodes = node_features.shape[0]

        # Make decoding reproducible. All randomness should be in the initial node_features:
        rng = np.random.default_rng(0)
        if connection_nodes is None:
            focus_nodes = rng.choice(num_nodes, size=beam_size, replace=num_nodes < beam_size)
        else:
            focus_nodes = rng.choice(
                connection_nodes, size=beam_size, replace=len(connection_nodes) < beam_size
            )
        beam = [
            Ray.construct_ray(
                idx=i,
                focus_node=focus_nodes[i],
                num_edge_types=self.num_edge_types_to_classify,
                node_types=node_types,
                add_self_loop_edges=self._use_self_loop_edges_in_partial_graphs,
                adjacency_lists=adjacency_lists,
                connection_nodes=connection_nodes,
                symmetrize_adjacency_lists=symmetrize_adjacency_lists,
                store_generation_trace=store_generation_traces,
            )
            for i in range(beam_size)
        ]

        # Run the beam decoding loop!
        while True:
            beam = self.one_beam_step(node_features, node_types, beam)
            if all(ray.finished for ray in beam):
                break

        return beam

    def one_beam_step(
        self, node_features: np.ndarray, node_types: List[str], beam: List[Ray]
    ) -> List[Ray]:
        """Calculate one update step for the beam decoder.

        For each ray given, this function calculates all possible one-step expansions. It then
        updates the set of rays using the best scoring (highest probability) extension choices. The
        number of rays in the returned dictionary is the same as the number of rays supplied.

        Args:
            node_features: a numpy array of node features for the graph to be decoded. This should
                have shape [num_nodes, embedding_dimension]
            node_types: a list of string representations of the atom types. The list should have
                length [num_nodes].
            beam: a list of Ray objects to be updated.

        Returns:
            A list of updated Ray objects.

        """
        num_nodes = node_features.shape[0]
        # There is only ever one graph given to the CGVAEDecoder at a time, so all nodes have idx 0.
        node_to_graph_map = np.zeros(shape=num_nodes, dtype=np.int32)
        # We will need this at multiple points throughout the loops:
        num_edge_types = len(beam[0].adjacency_lists)
        if self._use_self_loop_edges_in_partial_graphs:
            num_edge_types -= 1

        # We store all possible one-step extension choices in this list.
        beam_extension_choices = []

        # Iterate over all rays, calculating all extension choices for each of them.
        for ray in beam:
            # Easy out. This is a no-op choice on a finished ray.
            if ray.finished:
                beam_extension_choices.append(
                    RayExtension(
                        logprob=ray.logprob,
                        edge_choice=None,
                        edge_type=None,
                        ray_idx=ray.idx,
                        extension_type=ExtensionType.STOP_NODE,
                        generation_step_info=None,
                    )
                )
                continue

            # Find the valid edges that could be attached to the graph in this ray.
            valid_edges_raw = [
                [ray.focus_node, node_idx]
                for node_idx, state in ray.node_states.items()
                if state in {NodeState.DISCOVERED, NodeState.UNDISCOVERED}
                and not ray.contains_edge(source_idx=ray.focus_node, target_idx=node_idx)
            ]

            # We may need to strip out the self-loops for enforcing valence constraints:
            ray_no_self_loop_adjacency_lists = ray.adjacency_lists
            if self._use_self_loop_edges_in_partial_graphs:
                ray_no_self_loop_adjacency_lists = ray_no_self_loop_adjacency_lists[:-1]

            # Constrain the valid edges based on the atom valence.
            valid_edges = np.array(valid_edges_raw, dtype=np.int32)
            # Only worth further constraints if there are edges left to constrain:
            if valid_edges.size > 0:
                valid_edge_mask = constrain_edge_choices_based_on_valence(
                    start_node=ray.focus_node,
                    candidate_target_nodes=valid_edges[:, 1],
                    adjacency_lists=ray_no_self_loop_adjacency_lists,
                    node_types=node_types,
                )
                valid_edges = valid_edges[valid_edge_mask]

            # Deal with the easy case when there are no valid edges left for this focus node.
            if valid_edges.size == 0:
                beam_extension_choices.append(
                    RayExtension(
                        logprob=ray.logprob,
                        edge_choice=None,
                        edge_type=None,
                        ray_idx=ray.idx,
                        extension_type=ExtensionType.STOP_NODE,
                        generation_step_info=None,
                    )
                )
                continue

            # Calculate the node distance features:
            distance_features: List[int] = calculate_dist_from_focus_to_valid_target(
                adjacency_list=ray.adjacency_lists,
                focus_node=ray.focus_node,
                target_nodes=valid_edges[:, 1],
                symmetrise_adjacency_list=False,
            )
            # Reshape to make concatenation with topology features possible.
            distance_features = np.expand_dims(np.array(distance_features, dtype=np.float32), -1)
            # Calculate the topology featues:
            topology_features = calculate_topology_features(valid_edges, ray.molecule)

            edge_features = np.concatenate([distance_features, topology_features], axis=-1)

            # Calculate new node features:
            mol = ray.molecule
            concatenated_node_features = self._calculate_node_features(mol, node_features)

            # Run the GNN decoder to calculate edge choice and type logits.
            edge_choice_logits, edge_type_logits = self.__call__(
                CGVAEDecoderInput(
                    node_features=concatenated_node_features,
                    adjacency_lists=tuple(ray.adjacency_lists),
                    num_partial_graphs_in_batch=np.array(1, dtype=np.int32),
                    graph_to_focus_node_map=np.array([ray.focus_node], dtype=np.int32),
                    node_to_graph_map=node_to_graph_map,
                    valid_edge_choices=valid_edges,
                    edge_features=edge_features,
                ),
                training=False,
            )

            # Calculate edge choice and type probabilities from the logits, given the edge type
            # mask calculated based on node valence.
            valid_edge_type_masks = constrain_edge_types_based_on_valence(
                start_node=ray.focus_node,
                candidate_target_nodes=valid_edges[:, 1],
                adjacency_lists=ray_no_self_loop_adjacency_lists,
                node_types=node_types,
            )
            edge_choice_logprob, edge_type_logprob = self.calculate_logprobs_from_logits(
                edge_choice_logits, edge_type_logits, valid_edge_type_masks
            )

            # First, handle the case of choosing to create no edge (the last edge choice):
            stop_edge_logprob = edge_choice_logprob[-1]
            stop_logprob = ray.logprob + stop_edge_logprob

            # To optionally generate a decoder trace, we need to record all possible choice
            # together. To avoid going back and forth, we generate a MoleculeGenerationStepInfo
            # NamedTuple now, whose candidate_edge_infos is a (mutable) empty list which we will
            # fill with details in the loop below.
            if ray.generation_trace is not None:
                molecule_generation_step_info = MoleculeGenerationStepInfo(
                    focus_node_idx=ray.focus_node,
                    partial_molecule_adjacency_lists=ray_no_self_loop_adjacency_lists,
                    candidate_edge_infos=[],
                    no_edge_score=edge_choice_logits[-1].numpy()[0],
                    no_edge_logprob=stop_edge_logprob.numpy()[0],
                    no_edge_correct=None,
                )
            else:
                molecule_generation_step_info = None

            beam_extension_choices.append(
                RayExtension(
                    logprob=stop_logprob,
                    edge_choice=None,
                    edge_type=None,
                    ray_idx=ray.idx,
                    extension_type=ExtensionType.STOP_NODE,
                    generation_step_info=molecule_generation_step_info,
                )
            )

            # For each edge choice, and each edge type, calculate the correct BeamExtension.
            for edge_idx, valid_edge in enumerate(valid_edges):
                edge_logprob = edge_choice_logprob[edge_idx]
                edge_type_mask = valid_edge_type_masks[edge_idx]
                type_logprobs = edge_type_logprob[edge_idx]
                max_logprob = np.max(type_logprobs[tf.cast(edge_type_mask, dtype=tf.bool).numpy()])
                for type_idx in range(num_edge_types):
                    type_mask = edge_type_mask[type_idx]
                    if type_mask == 0:
                        continue
                    type_logprob = type_logprobs[type_idx]
                    if type_logprob != max_logprob:
                        continue
                    candidate_ray_logprob = ray.logprob + edge_logprob
                    beam_extension_choices.append(
                        RayExtension(
                            logprob=candidate_ray_logprob,
                            edge_choice=valid_edge,
                            edge_type=type_idx,
                            ray_idx=ray.idx,
                            extension_type=ExtensionType.ADD_EDGE,
                            generation_step_info=molecule_generation_step_info,
                            type_logprob=type_logprob,
                        )
                    )

                if molecule_generation_step_info is not None:
                    molecule_generation_step_info.candidate_edge_infos.append(
                        MoleculeGenerationEdgeCandidateInfo(
                            target_node_idx=valid_edge[1],
                            score=edge_choice_logits[edge_idx].numpy()[0],
                            logprob=edge_logprob.numpy()[0],
                            correct=None,
                            type_idx_to_logprobs=type_logprobs.numpy(),
                        )
                    )

        # Select the best extension choices, keeping the beam_size constant.
        beam_size = len(beam)
        best_extension_choices = heapq.nlargest(
            beam_size, beam_extension_choices, key=lambda x: x.logprob
        )
        return extend_beam(best_extension_choices, beam)

    def _calculate_node_features(self, mol, node_features):
        mol.UpdatePropertyCache()
        calculated_node_features = []
        for atom in mol.GetAtoms():
            atom_features = np.concatenate(
                [atom_featuriser.featurise(atom) for atom_featuriser in self._feature_extractors],
                axis=0,
            )
            calculated_node_features.append(atom_features)
        calculated_node_features = np.concatenate([calculated_node_features])
        concatenated_node_features = np.concatenate(
            [node_features, calculated_node_features], axis=-1
        )
        return concatenated_node_features.astype(np.float32)

    def calculate_logprobs_from_logits(
        self,
        edge_choice_logits: np.ndarray,
        edge_type_logits: np.ndarray,
        edge_type_mask: np.ndarray,
    ):
        """Calculate logprobs from logits, taking into account edge type masks.

        Args:
            edge_choice_logits: a numpy array of logits for the edge choices. Shape = [num_edges,]
            edge_type_logits: a numpy array of logits for the edge types for each edge choice.
                Shape = [num_edges, num_edge_types]
            edge_type_mask: a numpy array representing whether an edge type is valid for this graph
                or not. For edge e and type t, edge_type_mask[e, t] = 1 if that is a valid choice,
                and 0 otherwise. Shape = [num_edges, num_edge_types]

        Returns:
            A tuple of two numpy arrays. The first is the log probabilities for the edge choices.
            The second is the log probabilities for the edge types, given each edge choice.

        """
        # Easy one first:
        edge_choice_logprob = edge_choice_logits - np.log(np.sum(np.exp(edge_choice_logits)))

        # We want to multiply masked probabilities by 0. We are in log space, so instead we
        # subtract a large numer.
        scaled_edge_type_mask = (1 - edge_type_mask) * 1e7
        masked_edge_type_logits = edge_type_logits - scaled_edge_type_mask
        edge_type_log_denom = np.log(np.sum(np.exp(masked_edge_type_logits), axis=1, keepdims=True))
        edge_type_logprob = masked_edge_type_logits - edge_type_log_denom
        return edge_choice_logprob, edge_type_logprob

    def calculate_reconstruction_loss(
        self,
        node_type_logits: tf.Tensor,
        edge_logits: tf.Tensor,
        edge_type_logits: tf.Tensor,
        node_type_label_indices: tf.Tensor,
        num_graphs_in_batch: tf.Tensor,
        num_partial_graphs_in_batch: tf.Tensor,
        node_to_partial_graph_map: tf.Tensor,
        correct_target_node_multihot: tf.Tensor,
        valid_target_node_idx: tf.Tensor,
        stop_node_label: tf.Tensor,
        num_correct_edge_choices: tf.Tensor,
        one_hot_edge_types: tf.Tensor,
        valid_edge_types: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # Node classification.
        per_node_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=node_type_logits, labels=node_type_label_indices
        )  # Shape: [PV]
        if self._node_type_loss_weights is not None:
            per_node_loss_weight = tf.gather(
                params=self._node_type_loss_weights, indices=node_type_label_indices
            )  # Shape: [PV]
            per_node_losses *= per_node_loss_weight
        node_classification_loss = tf.reduce_sum(per_node_losses) / num_graphs_in_batch  # Shape: []

        # Edge selection.
        # We have some physically valid edge targets in valid_target_node_idx, and the correct
        # target nodes in correct_target_node_idx. We want to construct a tensor which is the same
        # shape as valid_target_node_idx which is 1 if that node idx is in correct_target_node_idx,
        # and 0 otherwise.
        correct_multihot = tf.concat(
            [correct_target_node_multihot, tf.cast(stop_node_label, tf.float32)], axis=0
        )  # Shape: [VE + PG]
        valid_logits = tf.squeeze(edge_logits)  # Shape: [VE + PG]
        valid_node_to_graph_map = tf.nn.embedding_lookup(
            node_to_partial_graph_map, valid_target_node_idx
        )  # Shape: [VE]
        valid_node_to_graph_map = tf.concat(
            [valid_node_to_graph_map, tf.range(0, num_partial_graphs_in_batch)], axis=0
        )  # Shape: [VE + PG]. The last PG elements are [0, ..., PG - 1]
        valid_edge_logprobs = traced_unsorted_segment_log_softmax(
            valid_logits, valid_node_to_graph_map, num_partial_graphs_in_batch
        )  # Shape: [VE + PG]

        # Compute the edge loss with the multihot objective.
        # For a single graph with three valid choices (+ stop node) of which two are correct,
        # we may have the following:
        #  valid_edge_logprobs = log([0.05, 0.5, 0.4, 0.05])
        #  num_correct_edge_choices = [2]
        #  correct_target_node_multihot = [0.0, 1.0, 1.0]
        #  correct_multihot = [0.0, 1.0, 1.0, 0.0]
        # To get the loss, we simply look at the things in valid_edge_logprobs that correspond
        # to correct entries.
        # However, to account for the _multi_hot nature, we scale up each entry of
        # valid_edge_logprobs by the number of correct choices, i.e., consider the
        # correct entries of
        #  log([0.05, 0.5, 0.4, 0.05]) + log([2, 2, 2, 2]) = log([0.1, 1.0, 0.8, 0.1])
        # In this form, we want to have each correct entry to be as near possible to 1.
        # Finally, we normalise loss contributions to by-graph, by dividing the crossentropy
        # loss by the number of correct choices (i.e., in the example above, this results in
        # a loss of -((log(1.0) + log(0.8)) / 2) = 0.11...).
        #
        # Note: num_correct_edge_choices does not include the choice of an edge to the stop
        # node, so can be zero.
        per_graph_num_correct_choices = tf.maximum(num_correct_edge_choices, 1)  # Shape: [PG]
        per_valid_edge_num_correct_choices = tf.cast(
            tf.nn.embedding_lookup(per_graph_num_correct_choices, valid_node_to_graph_map),
            tf.float32,
        )  # Shape: [VE]
        per_correct_edge_neglogprob = -(
            (valid_edge_logprobs + tf.math.log(per_valid_edge_num_correct_choices))
            * correct_multihot
            / per_valid_edge_num_correct_choices
        )  # Shape: [VE]
        edge_loss = tf.reduce_sum(per_correct_edge_neglogprob) / tf.cast(
            num_graphs_in_batch, tf.float32
        )  # Shape: []

        # Edge type loss.
        correct_target_indices = tf.cast(
            tf.squeeze(tf.where(correct_target_node_multihot != 0)), dtype=tf.int32
        )
        edge_type_logits_for_correct_edges = tf.gather(
            params=1 * edge_type_logits, indices=correct_target_indices
        )  # Shape: [CE, ET]
        # The `valid_edge_types` tensor is equal to 1 when the edge is valid, 0 otherwise.
        # We want to multiply the selection probabilities by this mask. Because the logits are in
        # log space, we instead subtract a large value from the logits wherever this mask is zero.
        scaled_edge_mask = (1 - tf.cast(valid_edge_types, dtype=tf.float32)) * tf.constant(
            1e7, dtype=tf.float32
        )  # Shape: [CE, ET]
        masked_edge_type_logits = (
            edge_type_logits_for_correct_edges - scaled_edge_mask
        )  # Shape: [CE, ET]
        edge_type_loss = tf.nn.softmax_cross_entropy_with_logits(
            one_hot_edge_types, masked_edge_type_logits
        )  # Shape: [CE]
        edge_type_loss = tf.reduce_sum(edge_type_loss) / num_graphs_in_batch  # Shape: []
        return edge_loss, edge_type_loss, node_classification_loss


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS)
