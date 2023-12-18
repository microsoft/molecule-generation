"""Decoding layer for a MoLeR."""
from typing import Any, Dict, List, Optional, Tuple, NamedTuple, Union, Iterable, Generator
import itertools
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from dpu_utils.tf2utils import MLP, unsorted_segment_log_softmax
from rdkit import Chem
from tf2_gnn.layers import (
    GNN,
    GNNInput,
    NodesToGraphRepresentationInput,
    WeightedSumGraphRepresentation,
)

from molecule_generation.chem.atom_feature_utils import (
    AtomFeatureExtractor,
    AtomTypeFeatureExtractor,
)
from molecule_generation.chem.molecule_dataset_utils import BOND_DICT
from molecule_generation.chem.rdkit_helpers import compute_canonical_atom_order, get_atom_symbol
from molecule_generation.chem.motif_utils import (
    find_motifs_from_vocabulary,
    MotifVocabulary,
)
from molecule_generation.utils.moler_decoding_utils import (
    DecoderSamplingMode,
    sample_indices_from_logprobs,
    restrict_to_beam_size_per_mol,
    MoLeRDecoderState,
    MoleculeGenerationAtomChoiceInfo,
    MoleculeGenerationAttachmentPointChoiceInfo,
    MoleculeGenerationEdgeChoiceInfo,
    MoleculeGenerationEdgeCandidateInfo,
)
from molecule_generation.utils.decoder_batching import batch_decoder_states

SMALL_NUMBER, BIG_NUMBER = 1e-7, 1e7
_COMPAT_DISTANCE_TRUNCATION = 100


@dataclass
class MoLeRDecoderMetrics:
    node_classification_loss: tf.Tensor
    first_node_classification_loss: tf.Tensor
    edge_loss: tf.Tensor
    edge_type_loss: tf.Tensor
    attachment_point_selection_loss: Optional[tf.Tensor]


@dataclass
class MoLeRDecoderOutput:
    node_type_logits: tf.Tensor
    edge_candidate_logits: tf.Tensor
    edge_type_logits: tf.Tensor
    attachment_point_selection_logits: tf.Tensor


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


def safe_divide_loss(loss: tf.Tensor, num_choices: tf.Tensor):
    """Divide `loss` by `num_choices`, but guard against `num_choices` being 0."""
    return loss / tf.maximum(tf.cast(num_choices, tf.float32), 1.0)


def compute_neglogprob_for_multihot_objective(
    logprobs: tf.Tensor,
    multihot_labels: tf.Tensor,
    per_decision_num_correct_choices: tf.Tensor,
) -> tf.Tensor:
    # Normalise by number of correct choices and mask out entries for wrong decisions:
    return -(
        (logprobs + tf.math.log(per_decision_num_correct_choices + SMALL_NUMBER))
        * multihot_labels
        / (per_decision_num_correct_choices + SMALL_NUMBER)
    )


class MoLeRDecoderInput(NamedTuple):
    node_features: tf.TensorArraySpec
    node_categorical_features: tf.TensorArraySpec
    adjacency_lists: Tuple[tf.Tensor, ...]
    num_graphs_in_batch: tf.Tensor
    node_to_graph_map: tf.Tensor
    graph_to_focus_node_map: tf.Tensor
    # Things that are used only if running with motifs:
    candidate_attachment_points: tf.Tensor
    # Additional things that are decoder, but not graph-related:
    input_molecule_representations: tf.Tensor
    graphs_requiring_node_choices: tf.Tensor
    candidate_edges: tf.Tensor
    candidate_edge_features: tf.Tensor


class MoLeRDecoder(tf.keras.layers.Layer):
    """Decode graph states using a combination of graph message passing layers and dense layers

    Example usage:
    >>> layer_input = MoLeRDecoderInput(
    ...     node_features=tf.random.normal(shape=(5, 12)),
    ...     node_categorical_features=tf.constant([], dtype=tf.int32),
    ...     adjacency_lists=(
    ...         tf.constant([[0, 1], [1, 2], [3, 4]], dtype=tf.int32),
    ...         tf.constant([[1, 2], [3, 4]], dtype=tf.int32),
    ...         tf.constant([[2, 0]], dtype=tf.int32),
    ...     ),
    ...     num_graphs_in_batch=tf.constant(2, dtype=tf.int32),
    ...     node_to_graph_map=tf.constant([0, 0, 0, 1, 1], dtype=tf.int32),
    ...     graph_to_focus_node_map=tf.constant([0, 3], dtype=tf.int32),
    ...     candidate_attachment_points=tf.constant([], dtype=tf.int32),
    ...     input_molecule_representations=tf.random.normal(shape=(2, 13)),
    ...     graphs_requiring_node_choices=[1],
    ...     candidate_edges=tf.constant([[0, 1], [0, 2], [3, 4]]),
    ...     candidate_edge_features=tf.constant([[0.6], [0.1], [0.9]])
    ... )
    >>> params = MoLeRDecoder.get_default_params()
    >>> from molecule_generation.chem.atom_feature_utils import AtomTypeFeatureExtractor
    >>> atom_type_featuriser = AtomTypeFeatureExtractor()
    >>> atom_type_featuriser.mark_metadata_initialised()
    >>> layer = MoLeRDecoder(
    ...     params,
    ...     atom_featurisers=[atom_type_featuriser],
    ...     index_to_node_type_map=atom_type_featuriser.index_to_atom_type_map
    ... )
    >>> output = layer(layer_input)
    >>> print(output)
    (<tf.Tensor... shape=(1, 2), dtype=float32, ...>, <tf.Tensor... shape=(5,), dtype=float32, ...>, <tf.Tensor... shape=(3, 3), dtype=float32, ...>, None)
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
            "gnn_dense_intermediate_layer_activation": "leaky_relu",
            "gnn_use_inter_layer_layernorm": True,
            "gnn_layer_input_dropout_rate": 0.0,
            "gnn_global_exchange_dropout_rate": 0.0,
            "graph_repr_size": 256,
            "graph_repr_mlp_layers": [256, 256],
            "graph_repr_num_heads": 16,
            "graph_repr_dropout_rate": 0.0,
            "categorical_features_embedding_dim": 64,
            "num_edge_types_to_classify": 3,
            "edge_candidate_scorer_hidden_layers": [128, 64, 32],
            "edge_type_selector_hidden_layers": [128, 64, 32],
            "edge_selection_dropout_rate": 0.0,
            "distance_truncation": 10,
            # When training a new model, this should probably be false.
            "compatible_distance_embedding_init": True,
            "node_type_selector_hidden_layers": [256, 256],
            "node_selection_dropout_rate": 0.0,
            "first_node_type_selector_hidden_layers": [256, 256],
            "first_node_type_selection_dropout_rate": 0.0,
            "attachment_point_selector_hidden_layers": [128, 64, 32],
            "attachment_point_selection_dropout_rate": 0.0,
            "max_nodes_per_batch": 10000,
        }
        decoder_gnn_params.update(these_params)
        return decoder_gnn_params

    def __init__(
        self,
        params,
        atom_featurisers: List[AtomFeatureExtractor],
        index_to_node_type_map: Dict[int, str],
        node_type_loss_weights: Optional[np.ndarray] = None,
        motif_vocabulary: Optional[MotifVocabulary] = None,
        node_categorical_num_classes: Optional[int] = None,
        **kwargs,
    ):
        """Initialise the layer."""
        super().__init__(**kwargs)
        if "max_nodes_per_batch" not in params:
            # Backwards-compatibility with old checkpoints that do not have "max_nodes_per_batch" parameter
            params["max_nodes_per_batch"] = self.get_default_params()["max_nodes_per_batch"]
        self._params = params
        self._num_edge_types_to_classify = params["num_edge_types_to_classify"]
        self._atom_featurisers = atom_featurisers

        self._index_to_node_type_map = index_to_node_type_map
        self._num_node_types = len(index_to_node_type_map)

        self._node_type_loss_weights = node_type_loss_weights
        self._motif_vocabulary = motif_vocabulary
        self._node_categorical_num_classes = node_categorical_num_classes

        if self.uses_motifs:
            # Record the set of atom types, which will be a subset of all node types.
            atom_type_featuriser: AtomTypeFeatureExtractor = next(
                featuriser for featuriser in atom_featurisers if featuriser.name == "AtomType"
            )

            self._atom_types = set(atom_type_featuriser.index_to_atom_type_map.values())

        # Layers to be built.
        self._gnn = GNN({k[4:]: v for k, v in params.items() if k.startswith("gnn_")})

        self._weighted_avg_of_nodes_to_graph_repr = WeightedSumGraphRepresentation(
            graph_representation_size=self._params["graph_repr_size"] // 2,
            num_heads=self._params["graph_repr_num_heads"] // 2,
            weighting_fun="softmax",
            scoring_mlp_layers=[l // 2 for l in self._params["graph_repr_mlp_layers"]],
            scoring_mlp_dropout_rate=self._params["graph_repr_dropout_rate"],
            transformation_mlp_layers=[l // 2 for l in self._params["graph_repr_mlp_layers"]],
            transformation_mlp_dropout_rate=self._params["graph_repr_dropout_rate"],
        )
        self._weighted_sum_of_nodes_to_graph_repr = WeightedSumGraphRepresentation(
            graph_representation_size=self._params["graph_repr_size"] // 2,
            num_heads=self._params["graph_repr_num_heads"] // 2,
            weighting_fun="sigmoid",
            scoring_mlp_layers=[l // 2 for l in self._params["graph_repr_mlp_layers"]],
            scoring_mlp_dropout_rate=self._params["graph_repr_dropout_rate"],
            transformation_mlp_layers=[l // 2 for l in self._params["graph_repr_mlp_layers"]],
            transformation_mlp_dropout_rate=self._params["graph_repr_dropout_rate"],
            transformation_mlp_result_upper_bound=5,
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
        self._node_type_selector = MLP(
            out_size=(self._num_node_types + 1),
            hidden_layers=self._params["node_type_selector_hidden_layers"],
            use_biases=True,
            dropout_rate=self._params["node_selection_dropout_rate"],
        )

        if self.has_first_node_type_selector:
            self._first_node_type_selector = MLP(
                out_size=(self._num_node_types),
                hidden_layers=self._params["first_node_type_selector_hidden_layers"],
                use_biases=True,
                dropout_rate=self._params["first_node_type_selection_dropout_rate"],
            )
        else:
            # Save the node type id corresponding to carbon, needed for decoding.
            self._carbon_type_idx = next(
                type_idx
                for type_idx, type_name in self._index_to_node_type_map.items()
                if type_name == "C"
            )

        # Create the attachment point selector only if running with motifs.
        if self.uses_motifs:
            self._attachment_point_selector = MLP(
                out_size=1,
                hidden_layers=self._params["attachment_point_selector_hidden_layers"],
                use_biases=True,
                dropout_rate=self._params["attachment_point_selection_dropout_rate"],
            )

        self._no_more_edges_representation: tf.Variable = None
        self._distance_embedding = None

        if self.uses_categorical_features:
            if "categorical_features_embedding_dim" in self._params:
                self._node_categorical_features_embedding = None
            else:
                # Older models use one hot vectors instead of dense embeddings, simulate that here.
                self._params["categorical_features_embedding_dim"] = node_categorical_num_classes
                self._node_categorical_features_embedding = np.eye(
                    node_categorical_num_classes, dtype=np.float32
                )

    def _batch_decoder_states(
        self, **kwargs
    ) -> Generator[Tuple[Dict[str, Any], List[MoLeRDecoderState]], None, None]:
        return batch_decoder_states(
            max_nodes_per_batch=self._params["max_nodes_per_batch"],
            atom_featurisers=self._atom_featurisers,
            motif_vocabulary=self._motif_vocabulary,
            uses_categorical_features=self.uses_categorical_features,
            **kwargs,
        )

    def compute_metrics(self, *, batch_features, batch_labels, task_output) -> MoLeRDecoderMetrics:
        node_classification_loss = self.compute_node_type_selection_loss(
            node_type_logits=task_output.node_type_logits,
            node_type_multihot_labels=batch_labels["correct_node_type_choices"],
        )

        first_node_classification_loss = self.compute_first_node_type_selection_loss(
            first_node_type_logits=task_output.first_node_type_logits,
            first_node_type_multihot_labels=batch_labels["correct_first_node_type_choices"],
        )

        edge_loss = self.compute_edge_candidate_selection_loss(
            num_graphs_in_batch=batch_features["num_partial_graphs_in_batch"],
            node_to_graph_map=batch_features["node_to_partial_graph_map"],
            candidate_edge_targets=batch_features["valid_edge_choices"][:, 1],
            edge_candidate_logits=task_output.edge_candidate_logits,
            per_graph_num_correct_edge_choices=batch_labels["num_correct_edge_choices"],
            edge_candidate_correctness_labels=batch_labels["correct_edge_choices"],
            no_edge_selected_labels=batch_labels["stop_node_label"],
        )

        edge_type_loss = self.compute_edge_type_selection_loss(
            valid_edge_types=batch_labels["valid_edge_types"],
            edge_type_logits=task_output.edge_type_logits,
            edge_candidate_correctness_labels=batch_labels["correct_edge_choices"],
            edge_type_onehot_labels=batch_labels["correct_edge_types"],
        )

        if self.uses_motifs:
            attachment_point_selection_loss = self.compute_attachment_point_selection_loss(
                num_graphs_in_batch=batch_features["num_partial_graphs_in_batch"],
                node_to_graph_map=batch_features["node_to_partial_graph_map"],
                attachment_point_selection_logits=task_output.attachment_point_selection_logits,
                attachment_point_candidate_choices=batch_features["valid_attachment_point_choices"],
                attachment_point_correct_choices=batch_labels["correct_attachment_point_choices"],
            )
        else:
            attachment_point_selection_loss = None

        return MoLeRDecoderMetrics(
            node_classification_loss=node_classification_loss,
            edge_loss=edge_loss,
            attachment_point_selection_loss=attachment_point_selection_loss,
            edge_type_loss=edge_type_loss,
            first_node_classification_loss=first_node_classification_loss,
        )

    @property
    def num_edge_types_to_classify(self) -> int:
        return self._num_edge_types_to_classify

    @property
    def has_first_node_type_selector(self) -> bool:
        # Older models may lack the first node type selector.
        return "first_node_type_selector_hidden_layers" in self._params

    @property
    def uses_motifs(self) -> bool:
        return self._motif_vocabulary is not None

    @property
    def uses_categorical_features(self) -> bool:
        return self._node_categorical_num_classes is not None

    def build(self, tensor_shapes: MoLeRDecoderInput):
        # We extend the initial node features by an is-focus-node-bit, and possibly also embedding
        # of categorical features, so prepare that shape here:
        focus_node_bit_size = 1
        node_features_dim = tensor_shapes.node_features[-1] + focus_node_bit_size

        if self.uses_categorical_features:
            node_features_dim += self._params["categorical_features_embedding_dim"]

        initial_node_feature_shape = tf.TensorShape(dims=(None, node_features_dim))
        input_molecule_representation_size = tensor_shapes.input_molecule_representations[-1]

        with tf.name_scope("decoder_gnn"):
            # No need to generalise input shapes here. Taken care of by GNN layer.
            self._gnn.build(
                GNNInput(
                    node_features=initial_node_feature_shape,
                    adjacency_lists=tensor_shapes.adjacency_lists,
                    node_to_graph_map=tensor_shapes.node_to_graph_map,
                    num_graphs=tensor_shapes.num_graphs_in_batch,
                )
            )

        # We get the initial GNN input (after projection) + results for all layers:
        node_repr_size = self._params["gnn_hidden_dim"] * (1 + self._params["gnn_num_layers"])

        # Build the individual layers, which we've initialised in __init__():
        node_to_graph_repr_input = NodesToGraphRepresentationInput(
            node_embeddings=tf.TensorShape((None, node_repr_size)),
            node_to_graph_map=tensor_shapes.node_to_graph_map,
            num_graphs=tensor_shapes.num_graphs_in_batch,
        )
        with tf.name_scope("graph_representation_computation"):
            with tf.name_scope("weighted_avg"):
                self._weighted_avg_of_nodes_to_graph_repr.build(node_to_graph_repr_input)
            with tf.name_scope("weighted_sum"):
                self._weighted_sum_of_nodes_to_graph_repr.build(node_to_graph_repr_input)

        # Edge candidates are represented by a graph-global representation,
        # the representations of source and target nodes, and edge features:
        edge_feature_dim = tensor_shapes.candidate_edge_features[-1]
        edge_candidate_representation_size = (
            input_molecule_representation_size
            + self._params["graph_repr_size"]
            + 2 * node_repr_size
            + edge_feature_dim
        )
        with tf.name_scope("edge_candidate_scorer"):
            self._edge_candidate_scorer.build(
                tf.TensorShape((None, edge_candidate_representation_size))
            )
            self._no_more_edges_representation = self.add_weight(
                name="no_more_edges_representation",
                shape=(1, node_repr_size + edge_feature_dim),
                trainable=True,
            )

        with tf.name_scope("edge_type_selector"):
            self._edge_type_selector.build(input_shape=[None, edge_candidate_representation_size])

        with tf.name_scope("node_type_selector"):
            self._node_type_selector.build(
                input_shape=[
                    None,
                    self._params["graph_repr_size"] + input_molecule_representation_size,
                ]
            )

        if self.has_first_node_type_selector:
            with tf.name_scope("first_node_type_selector"):
                self._first_node_type_selector.build(
                    input_shape=[None, input_molecule_representation_size]
                )

        if self.uses_motifs:
            with tf.name_scope("attachment_point_selector"):
                self._attachment_point_selector.build(
                    input_shape=[
                        None,
                        self._params["graph_repr_size"]
                        + input_molecule_representation_size
                        + node_repr_size,
                    ]
                )

        if self.uses_categorical_features and self._node_categorical_features_embedding is None:
            with tf.name_scope("node_categorical_features_embedding"):
                self._node_categorical_features_embedding = self.add_weight(
                    name="categorical_features_embedding",
                    shape=(
                        self._node_categorical_num_classes,
                        self._params["categorical_features_embedding_dim"],
                    ),
                    trainable=True,
                )

        with tf.name_scope("distance_embedding"):
            self._distance_embedding = self.add_weight(
                name="distance_embedding",
                shape=(self._params.get("distance_truncation", _COMPAT_DISTANCE_TRUNCATION), 1),
                trainable=True,
            )

        super().build(tensor_shapes)

    def call(self, input: MoLeRDecoderInput, training: bool = False) -> MoLeRDecoderOutput:
        """
        Call graph generation model for training and batch evaluation purposes, using
        teacher forcing to a correct choice to efficiently batch training.

        Note: At inference time, the methods pick_node_type and pick_edge should be
        used directly.

        Shape abbreviations used throughout the model:
        - PG = number of _p_artial _g_raphs
        - PV = number of _p_artial graph _v_ertices
        - PD = size of _p_artial graph _representation _d_imension
        - VD = GNN _v_ertex representation _d_imension
        - MD = _m_olecule representation _d_imension
        - EFD = _e_dge _f_eature _d_imension
        - NTP = number of partial graphs requiring a _n_ode _t_ype _p_ick
        - NT = number of _n_ode _t_ypes
        - CE = number of _c_andidate _e_dges
        - CCE = number of _c_orrect _c_andidate _e_dges
        - ET = number of _e_dge _t_ypes
        - CA = number of _c_andidate _a_ttachment points
        - AP = number of _a_ttachment _p_oint choices
        """
        # Because pick_node_type and pick_edge share operations on the partial graphs, and
        # we want to avoid duplicating them during training, we do this first:
        graph_representations, node_representations = self.calculate_node_and_graph_representations(
            node_features=input.node_features,
            node_categorical_features=input.node_categorical_features,
            adjacency_lists=input.adjacency_lists,
            num_graphs_in_batch=input.num_graphs_in_batch,
            node_to_graph_map=input.node_to_graph_map,
            graph_to_focus_node_map=input.graph_to_focus_node_map,
            candidate_attachment_points=input.candidate_attachment_points,
            training=training,
        )
        node_type_logits = self.pick_node_type(
            input_molecule_representations=input.input_molecule_representations,
            graph_representations=graph_representations,
            graphs_requiring_node_choices=input.graphs_requiring_node_choices,
            training=training,
        )
        edge_candidate_logits, edge_type_logits = self.pick_edge(
            input_molecule_representations=input.input_molecule_representations,
            graph_representations=graph_representations,
            node_representations=node_representations,
            num_graphs_in_batch=input.num_graphs_in_batch,
            graph_to_focus_node_map=input.graph_to_focus_node_map,
            node_to_graph_map=input.node_to_graph_map,
            candidate_edge_targets=input.candidate_edges[:, 1],
            candidate_edge_features=input.candidate_edge_features,
            training=training,
        )

        if self.uses_motifs:
            attachment_point_selection_logits = self.pick_attachment_point(
                input_molecule_representations=input.input_molecule_representations,
                graph_representations=graph_representations,
                node_representations=node_representations,
                node_to_graph_map=input.node_to_graph_map,
                candidate_attachment_points=input.candidate_attachment_points,
                training=training,
            )
        else:
            attachment_point_selection_logits = None

        return MoLeRDecoderOutput(
            node_type_logits=node_type_logits,
            edge_candidate_logits=edge_candidate_logits,
            edge_type_logits=edge_type_logits,
            attachment_point_selection_logits=attachment_point_selection_logits,
        )

    def calculate_node_and_graph_representations(
        self,
        node_features: tf.Tensor,
        node_categorical_features: tf.Tensor,
        adjacency_lists: Tuple[tf.Tensor, ...],
        num_graphs_in_batch: tf.Tensor,
        node_to_graph_map: tf.Tensor,
        graph_to_focus_node_map: tf.Tensor,
        candidate_attachment_points: tf.Tensor,
        training: bool = False,
    ):
        if self.uses_categorical_features:
            embedded_categorical_features = tf.nn.embedding_lookup(
                self._node_categorical_features_embedding, node_categorical_features
            )

            initial_node_features = tf.concat(
                [node_features, embedded_categorical_features], axis=-1
            )
        else:
            initial_node_features = node_features

        if self.uses_motifs:
            # We will add the in-focus bit to both the actual focus nodes and nodes for attachment point
            # selection. Note that the ground-truth attachment point is present in
            # `graph_to_focus_node_map`; if we didn't also mark all valid attachment points, the
            # ground-truth answer would leak into the data.
            nodes_to_set_in_focus_bit = tf.concat(
                [graph_to_focus_node_map, candidate_attachment_points], axis=0
            )
        else:
            nodes_to_set_in_focus_bit = graph_to_focus_node_map

        node_is_in_focus_bit = tf.scatter_nd(
            indices=tf.expand_dims(nodes_to_set_in_focus_bit, axis=-1),
            updates=tf.ones(shape=(tf.shape(nodes_to_set_in_focus_bit)[0], 1), dtype=tf.float32),
            shape=(tf.shape(node_features)[0], 1),
        )

        if self.uses_motifs:
            # For graphs which require choosing an attachment point, one of the choices is marked as a
            # focus node. The line above will then add `1` to its position twice - correcting it here.
            node_is_in_focus_bit = tf.minimum(node_is_in_focus_bit, 1.0)

        initial_node_features = tf.concat([initial_node_features, node_is_in_focus_bit], axis=-1)

        # Encode the graphs we are extending using a GNN:
        encoder_input = GNNInput(
            node_features=initial_node_features,
            adjacency_lists=adjacency_lists,
            node_to_graph_map=node_to_graph_map,
            num_graphs=num_graphs_in_batch,
        )
        _, node_representations_at_all_layers = self._gnn(
            encoder_input, training=training, return_all_representations=True
        )
        node_representations = tf.concat(
            node_representations_at_all_layers, axis=-1
        )  # Shape [V, VD*(num_layers+1)]

        # Also calculate graph-level representations of each graph:
        graph_representation_layer_input = NodesToGraphRepresentationInput(
            node_embeddings=node_representations,
            node_to_graph_map=node_to_graph_map,
            num_graphs=num_graphs_in_batch,
        )
        weighted_avg_graph_repr = self._weighted_avg_of_nodes_to_graph_repr(
            graph_representation_layer_input, training=training
        )
        weighted_sum_graph_repr = self._weighted_sum_of_nodes_to_graph_repr(
            graph_representation_layer_input, training=training
        )
        graph_representations = tf.concat(
            [weighted_avg_graph_repr, weighted_sum_graph_repr], axis=-1
        )  # shape: [PG, GD]
        return graph_representations, node_representations

    def pick_node_type(
        self,
        input_molecule_representations: tf.Tensor,
        graph_representations: tf.Tensor,
        graphs_requiring_node_choices: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Pick the node type of the next node, given

        Args:
            input_molecule_representations: the "target" representation for the partial graph,
                i.e., the input informing the decoder on what it should produce.
                Shape: [PG, MD]
            graph_representations: the representations of the partial graphs we try to extend
                with a fresh node.
                Shape: [PG, PD]
            graphs_requiring_node_choices: ids of the graphs for which we need to make node
                choices.
                Shape: [NTP], with NTP <= PG
            training: A bool denoting whether this is a training call.

        Returns:
            A tensor which represents node classification logits, which has shape [NTP, NT + 1].
            The first NT columns represent the logits for the node types in the graph, and the
            final column represents the logit for the special "stop atom," which signals that the
            graph is complete.
        """
        original_and_calculated_graph_representations = tf.concat(
            [
                tf.gather(input_molecule_representations, graphs_requiring_node_choices),
                tf.gather(graph_representations, graphs_requiring_node_choices),
            ],
            axis=-1,
        )  # Shape: [NTP, MD + PD]

        return self._node_type_selector(
            original_and_calculated_graph_representations, training=training
        )

    def pick_first_node_type(
        self,
        input_molecule_representations: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Pick the first node type of the next node, given

        Args:
            input_molecule_representations: the "target" representation for the partial graph,
                i.e., the input informing the decoder on what it should produce.
                Shape: [PG, MD]
            training: A bool denoting whether this is a training call.

        Returns:
            A tensor which represents first node classification logits, which have shape [PG, NT].
            The columns represent the logits for the first node type in each input graph.
        """
        return self._first_node_type_selector(input_molecule_representations, training=training)

    def pick_edge(
        self,
        input_molecule_representations: tf.Tensor,
        graph_representations: tf.Tensor,
        node_representations: tf.Tensor,
        num_graphs_in_batch: tf.Tensor,
        graph_to_focus_node_map: tf.Tensor,
        node_to_graph_map: tf.Tensor,
        candidate_edge_targets: tf.Tensor,
        candidate_edge_features: tf.Tensor,
        training: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Given candidate edges in partial graphs, compute logits for the likelihood of adding these
        candidates, as well as logits for the type of the edge if it is picked.

        Args:
            input_molecule_representations: the "target" representation for the partial graph,
                i.e., the input informing the decoder on what it should produce.
                Shape: [PG, MD]
            graph_representations: the representations of the partial graphs we operate on.
                Shape: [PG, PD]
            node_representations: the representations of individual nodes in all partial
                graphs we operate on.
                Shape: [PV, VD]
            num_graphs_in_batch: the number of partial graphs in the batch. Should be
                equal to PG.
            graph_to_focus_node_map: the index of the focus node for that partial graph.
                Shape: [PG,]
            node_to_graph_map: the index of the partial graph that this node came from.
                Shape: [PV,], with values in the range [0, ..., PG - 1]
            candidate_edge_targets: the edge targets which are allowable at this point.
                The corresponding source of the edge is always the focus node of the
                partial graph containing a potential target. Shape: [CE]
            candidate_edge_features: features associated with the candidate edges. A tensor
                of rank 2, where the first dimension matches the number of valid edges, and
                the second is the number of edges features. Shape: [CE, EFD]
            training: A bool denoting whether this is a training call.

        Returns:
            A tuple of two tensors. The first represents the edge choice logits, which has shape
            [CE + PG]. The first CE elements of the array correspond to the logits for the
            focus_node -> valid edge to be added. The final PG elements correspond to the logits
            for the stop node for each of the PG partial graphs (in order).
            The second represents the edge type logits, which has shape [CE, ET]
        """
        # Gather up focus node representations:
        focus_node_representations = tf.gather(
            node_representations, graph_to_focus_node_map
        )  # shape: [PG, VD*(num_layers+1)]

        graph_and_focus_node_representations = tf.concat(
            [input_molecule_representations, graph_representations, focus_node_representations],
            axis=-1,
        )  # shape: [PG, MD + PD + VD*(num_layers+1)]

        # We have to do a couple of gathers here ensure that we decode only the valid nodes.
        valid_target_to_graph_map = tf.gather(
            node_to_graph_map, candidate_edge_targets
        )  # shape: [CE]

        graph_and_focus_node_representations_per_edge_candidate = tf.gather(
            graph_and_focus_node_representations, valid_target_to_graph_map
        )  # shape: [CE, MD + PD + VD*(num_layers+1)]

        # Extract the features for the valid edges only.
        edge_candidate_target_node_representations = tf.gather(
            node_representations, candidate_edge_targets
        )  # shape: [CE, VD*(num_layers+1)]

        # The zeroth element of edge_features is the graph distance. We need to look that up
        # in the distance embeddings:
        truncated_distances = tf.minimum(
            tf.cast(candidate_edge_features[:, 0], dtype=np.int32),
            self._params.get("distance_truncation", _COMPAT_DISTANCE_TRUNCATION) - 1,
        )  # shape: [CE]
        distance_embedding = tf.nn.embedding_lookup(
            self._distance_embedding, truncated_distances
        )  # shape: [CE, 1]
        # Concatenate all the node features, to form focus_node -> target_node edge features
        edge_candidate_representation = tf.concat(
            [
                graph_and_focus_node_representations_per_edge_candidate,
                edge_candidate_target_node_representations,
                distance_embedding,
                candidate_edge_features[:, 1:],
            ],
            axis=-1,
        )  # shape: [CE, MD + PD + 2 * VD*(num_layers+1) + FD]

        # Calculate the stop node features as well.
        stop_edge_selection_representation = tf.concat(
            [
                graph_and_focus_node_representations,
                tf.tile(
                    self._no_more_edges_representation,
                    multiples=(num_graphs_in_batch, 1),
                ),
            ],
            axis=-1,
        )  # shape: [PG, MD + PD + 2 * VD*(num_layers+1) + FD]

        edge_candidate_and_stop_features = tf.concat(
            [edge_candidate_representation, stop_edge_selection_representation], axis=0
        )  # shape: [CE + PG, MD + PD + 2 * VD*(num_layers+1) + FD]

        edge_candidate_logits = tf.squeeze(
            self._edge_candidate_scorer(edge_candidate_and_stop_features, training=training),
            axis=-1,
        )  # shape: [CE + PG]
        edge_type_logits = self._edge_type_selector(
            edge_candidate_representation, training=training
        )  # shape: [CE, ET]

        return edge_candidate_logits, edge_type_logits

    def pick_attachment_point(
        self,
        input_molecule_representations: tf.Tensor,
        graph_representations: tf.Tensor,
        node_representations: tf.Tensor,
        node_to_graph_map: tf.Tensor,
        candidate_attachment_points: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Given candidate attachment points in partial graphs, compute logits for the likelihood of
        choosing them.

        Args:
            input_molecule_representations: the "target" representation for the partial graph,
                i.e., the input informing the decoder on what it should produce.
                Shape: [PG, MD]
            graph_representations: the representations of the partial graphs we operate on.
                Shape: [PG, PD]
            node_representations: the representations of individual nodes in all partial
                graphs we operate on.
                Shape: [PV, VD]
            node_to_graph_map: the index of the partial graph that this node came from.
                Shape: [PV,], with values in the range [0, ..., PG - 1]
            candidate_attachment_points: attachment point candidates in all partial graphs.
                Shape: [CA], with values in the range [0, ..., PV - 1]
            training: A bool denoting whether this is a training call.

        Returns:
            A tensor representing the attachment point selection logits, which has shape [CA].
        """
        # This method should never be called unless we are running with motifs.
        assert self.uses_motifs

        original_and_calculated_graph_representations = tf.concat(
            [input_molecule_representations, graph_representations],
            axis=-1,
        )  # Shape: [PG, MD + PD]

        # Map attachment point candidates to their respective partial graphs.
        partial_graphs_for_attachment_point_choices = tf.gather(
            node_to_graph_map, candidate_attachment_points
        )  # Shape: [CA]

        # To score an attachment point, we condition on the representations of input and partial
        # graphs, along with the representation of the attachment point candidate in question.
        attachment_point_representations = tf.concat(
            [
                tf.gather(
                    original_and_calculated_graph_representations,
                    partial_graphs_for_attachment_point_choices,
                ),
                tf.gather(1 * node_representations, candidate_attachment_points),
            ],
            axis=-1,
        )  # Shape: [CA, MD + PD + VD*(num_layers+1)]

        attachment_point_selection_logits = tf.squeeze(
            self._attachment_point_selector(attachment_point_representations, training=training),
            axis=-1,
        )  # Shape: [CA]

        return attachment_point_selection_logits

    @tf.function(experimental_relax_shapes=True)
    def compute_node_type_selection_loss(
        self,
        node_type_logits: tf.Tensor,
        node_type_multihot_labels: tf.Tensor,
    ) -> tf.Tensor:
        # We have only made predictions for a subset of NTP graphs, but our labels also only
        # cover these.
        # Complication 1: We have NT labels, but NT + 1 logits, where the final logit corresponds
        #   to the "no node to be added case". We need to split this out explicitly.
        # Complication 2: As we have a multihot objective, we can't use straight softmax
        #   cross-entropy, but need to build this ourselves, accounting for the number of correct
        #   choices (see below for an explanation with example for edge choices):
        per_node_decision_logprobs = tf.nn.log_softmax(
            node_type_logits, axis=-1
        )  # Shape: [NTP, NT + 1]
        per_node_decision_num_correct_choices = tf.math.reduce_sum(
            node_type_multihot_labels, keepdims=True, axis=-1
        )  # Shape [NTP, 1]

        per_correct_node_decision_normalised_neglogprob = compute_neglogprob_for_multihot_objective(
            logprobs=per_node_decision_logprobs[:, :-1],
            multihot_labels=node_type_multihot_labels,
            per_decision_num_correct_choices=per_node_decision_num_correct_choices,
        )  # Shape [NTP, NT]

        no_node_decision_correct = tf.math.equal(
            per_node_decision_num_correct_choices, 0.0
        )  # Shape [NTP]
        per_correct_no_node_decision_neglogprob = -(
            per_node_decision_logprobs[:, -1]
            * tf.cast(tf.squeeze(no_node_decision_correct), tf.float32)
        )  # Shape [NTP]

        if self._node_type_loss_weights is not None:
            per_correct_node_decision_normalised_neglogprob *= self._node_type_loss_weights[:-1]
            per_correct_no_node_decision_neglogprob *= self._node_type_loss_weights[-1]

        # Loss is the sum of the masked (no) node decisions, averaged over number of decisions made:
        total_node_type_loss = tf.reduce_sum(
            per_correct_node_decision_normalised_neglogprob
        ) + tf.reduce_sum(per_correct_no_node_decision_neglogprob)
        node_type_loss = safe_divide_loss(
            total_node_type_loss, tf.shape(node_type_multihot_labels)[0]
        )

        return node_type_loss

    @tf.function(experimental_relax_shapes=True)
    def compute_first_node_type_selection_loss(
        self,
        first_node_type_logits: tf.Tensor,
        first_node_type_multihot_labels: tf.Tensor,
    ) -> tf.Tensor:
        per_graph_logprobs = tf.nn.log_softmax(first_node_type_logits, axis=-1)  # Shape: [PG, NT]
        per_graph_num_correct_choices = tf.math.reduce_sum(
            first_node_type_multihot_labels, keepdims=True, axis=-1
        )  # Shape [PG, 1]

        per_graph_normalised_neglogprob = compute_neglogprob_for_multihot_objective(
            logprobs=per_graph_logprobs,
            multihot_labels=first_node_type_multihot_labels,
            per_decision_num_correct_choices=per_graph_num_correct_choices,
        )  # Shape [PG, NT]

        if self._node_type_loss_weights is not None:
            per_graph_normalised_neglogprob *= self._node_type_loss_weights[:-1]

        first_node_type_loss = safe_divide_loss(
            tf.reduce_sum(per_graph_normalised_neglogprob),
            tf.shape(first_node_type_multihot_labels)[0],
        )

        return first_node_type_loss

    @tf.function(experimental_relax_shapes=True)
    def compute_edge_candidate_selection_loss(
        self,
        num_graphs_in_batch: tf.Tensor,
        node_to_graph_map: tf.Tensor,
        candidate_edge_targets: tf.Tensor,
        edge_candidate_logits: tf.Tensor,
        per_graph_num_correct_edge_choices: tf.Tensor,
        edge_candidate_correctness_labels: tf.Tensor,
        no_edge_selected_labels: tf.Tensor,
    ) -> tf.Tensor:
        # First, we construct full labels for all edge decisions, which are the concat of
        # edge candidate logits and the logits for choosing no edge:
        edge_correctness_labels = tf.concat(
            [edge_candidate_correctness_labels, tf.cast(no_edge_selected_labels, tf.float32)],
            axis=0,
        )  # Shape: [CE + PG]

        # To compute a softmax over all candidate edges (and the "no edge" choice) corresponding
        # to the same graph, we first need to build the map from each logit to the corresponding
        # graph id. Then, we can do an unsorted_segment_softmax using that map:
        edge_candidate_to_graph_map = tf.gather(
            node_to_graph_map, candidate_edge_targets
        )  # Shape: [CE]
        edge_candidate_to_graph_map = tf.concat(
            [edge_candidate_to_graph_map, tf.range(0, num_graphs_in_batch)], axis=0
        )  # Shape: [CE + PG]. The last PG elements are [0, ..., PG - 1]
        edge_candidate_logprobs = traced_unsorted_segment_log_softmax(
            logits=edge_candidate_logits,
            segment_ids=edge_candidate_to_graph_map,
            num_segments=num_graphs_in_batch,
        )  # Shape: [CE + PG]

        # Compute the edge loss with the multihot objective.
        # For a single graph with three valid choices (+ stop node) of which two are correct,
        # we may have the following:
        #  edge_candidate_logprobs = log([0.05, 0.5, 0.4, 0.05])
        #  per_graph_num_correct_edge_choices = [2]
        #  edge_candidate_correctness_labels = [0.0, 1.0, 1.0]
        #  edge_correctness_labels = [0.0, 1.0, 1.0, 0.0]
        # To get the loss, we simply look at the things in edge_candidate_logprobs that correspond
        # to correct entries.
        # However, to account for the _multi_hot nature, we scale up each entry of
        # edge_candidate_logprobs by the number of correct choices, i.e., consider the
        # correct entries of
        #  log([0.05, 0.5, 0.4, 0.05]) + log([2, 2, 2, 2]) = log([0.1, 1.0, 0.8, 0.1])
        # In this form, we want to have each correct entry to be as near possible to 1.
        # Finally, we normalise loss contributions to by-graph, by dividing the crossentropy
        # loss by the number of correct choices (i.e., in the example above, this results in
        # a loss of -((log(1.0) + log(0.8)) / 2) = 0.11...).

        # Note: per_graph_num_correct_edge_choices does not include the choice of an edge to
        # the stop node, so can be zero.
        per_graph_num_correct_edge_choices = tf.maximum(
            per_graph_num_correct_edge_choices, 1
        )  # Shape: [PG]
        per_edge_candidate_num_correct_choices = tf.nn.embedding_lookup(
            tf.cast(per_graph_num_correct_edge_choices, tf.float32), edge_candidate_to_graph_map
        )  # Shape: [CE]
        per_correct_edge_neglogprob = -(
            (edge_candidate_logprobs + tf.math.log(per_edge_candidate_num_correct_choices))
            * edge_correctness_labels
            / per_edge_candidate_num_correct_choices
        )  # Shape: [CE]

        # Normalise by number of graphs for which we made edge selection decisions:
        edge_loss = safe_divide_loss(
            tf.reduce_sum(per_correct_edge_neglogprob), num_graphs_in_batch
        )

        return edge_loss

    @tf.function(experimental_relax_shapes=True)
    def compute_edge_type_selection_loss(
        self,
        valid_edge_types: tf.Tensor,
        edge_type_logits: tf.Tensor,
        edge_candidate_correctness_labels: tf.Tensor,
        edge_type_onehot_labels: tf.Tensor,
    ) -> tf.Tensor:
        correct_target_indices = tf.cast(
            tf.squeeze(tf.where(edge_candidate_correctness_labels != 0)), dtype=tf.int32
        )  # Shape: [CCE]
        edge_type_logits_for_correct_edges = tf.gather(
            params=1 * edge_type_logits, indices=correct_target_indices
        )  # Shape: [CCE, ET]

        # The `valid_edge_types` tensor is equal to 1 when the edge is valid (it may be invalid due
        # to valency constraints), 0 otherwise.
        # We want to multiply the selection probabilities by this mask. Because the logits are in
        # log space, we instead subtract a large value from the logits wherever this mask is zero.
        scaled_edge_mask = (1 - tf.cast(valid_edge_types, dtype=tf.float32)) * tf.constant(
            BIG_NUMBER, dtype=tf.float32
        )  # Shape: [CCE, ET]
        masked_edge_type_logits = (
            edge_type_logits_for_correct_edges - scaled_edge_mask
        )  # Shape: [CCE, ET]
        edge_type_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=edge_type_onehot_labels, logits=masked_edge_type_logits
        )  # Shape: [CCE]

        # Normalise by the number of edges for which we needed to pick a type:
        edge_type_loss = safe_divide_loss(
            tf.reduce_sum(edge_type_loss), tf.shape(edge_type_loss)[0]
        )

        return edge_type_loss

    @tf.function(experimental_relax_shapes=True)
    def compute_attachment_point_selection_loss(
        self,
        num_graphs_in_batch: tf.Tensor,
        node_to_graph_map: tf.Tensor,
        attachment_point_selection_logits: tf.Tensor,
        attachment_point_candidate_choices: tf.Tensor,
        attachment_point_correct_choices: tf.Tensor,
    ) -> tf.Tensor:
        # This method should never be called unless are running with motifs.
        assert self.uses_motifs

        attachment_point_candidate_to_graph_map = tf.gather(
            node_to_graph_map, attachment_point_candidate_choices
        )  # Shape: [CA]

        # Compute log softmax of the logits within each partial graph.
        attachment_point_candidate_logprobs = (
            traced_unsorted_segment_log_softmax(
                logits=attachment_point_selection_logits,
                segment_ids=attachment_point_candidate_to_graph_map,
                num_segments=num_graphs_in_batch,
            )
            * 1.0
        )  # Shape: [CA]

        attachment_point_correct_choice_neglogprobs = -tf.gather(
            attachment_point_candidate_logprobs, attachment_point_correct_choices
        )  # Shape: [AP]

        attachment_point_selection_loss = safe_divide_loss(
            tf.reduce_sum(attachment_point_correct_choice_neglogprobs),
            tf.shape(attachment_point_correct_choice_neglogprobs)[0],
        )

        return attachment_point_selection_loss

    def _is_atom_type(self, node_type: str):
        if not self.uses_motifs:
            return True
        else:
            return node_type in self._atom_types

    def _add_atom_or_motif(
        self,
        decoder_state: MoLeRDecoderState,
        node_type: str,
        logprob: float,
        choice_info: Optional[
            Union[MoleculeGenerationAtomChoiceInfo, MoleculeGenerationAttachmentPointChoiceInfo]
        ],
    ) -> Tuple[MoLeRDecoderState, bool]:
        # If we are running with motifs, we need to check whether `node_type` is an atom or a motif.
        if self._is_atom_type(node_type):
            # print(I {decoder_state.molecule_id} {decoder_state.logprob:12f}: Adding new atom {node_type} - p={logprob:5f}")
            return (
                MoLeRDecoderState.new_with_added_atom(
                    decoder_state,
                    node_type,
                    atom_logprob=logprob,
                    atom_choice_info=choice_info,
                ),
                False,
            )
        else:
            # print(I {decoder_state.molecule_id} {decoder_state.logprob:12f}: Adding motif {node_type} - p={logprob:5f}")
            return (
                MoLeRDecoderState.new_with_added_motif(
                    decoder_state,
                    node_type,
                    motif_logprob=logprob,
                    atom_choice_info=choice_info,
                ),
                True,
            )

    def decode(
        self,
        graph_representations: List[Union[tf.Tensor, np.ndarray]],
        initial_molecules: Optional[List[Optional[Chem.Mol]]] = None,
        mol_ids: Optional[Iterable[Any]] = None,
        store_generation_traces: bool = False,
        max_num_steps: int = 120,
        beam_size: int = 1,
        sampling_mode: DecoderSamplingMode = DecoderSamplingMode.GREEDY,
    ) -> List[MoLeRDecoderState]:
        """Decoding procedure for MoLeR.

        This method can handle generation of many graphs in parallel and implements greedy
        decoding. Roughly, the following algorithm is implemented:

        while True:
            new_atom_or_motif = pick_atom_or_motif(G, sampled_input_representation)
            if new_atom_or_motif == END_GEN:
                break

            G.add_atom_or_motif(new_atom_or_motif)
            focus_atom = pick_attachment_point(new_atom_or_motif)

            while True:
                new_bond = pick_bond(focus_atom, G, sampled_input_representation)
                if new_bond == END_BONDS:
                    break
                G.add_bond(new_bond)

        Args:
            graph_representations: List of representations used to condition the
                decoding procedure.
            initial_molecules: List of initial molecules (or scaffolds), each representing
                a partial graph that will definitely be contained in the decoded molecule.
                Has to have the same length as `graph_representations`, but can contain
                `None`, in which case no subgraph is guaranteed (generation proceeds from
                scratch). Setting `initial_molecules` to `None` is equivalent to providing
                a list full of `None`s.
            mol_ids: Iterable of IDs associated with the input molecule representations.
                These will simply be propagated throughout the process, allowing to
                connect outputs to the inputs, as we are changing the order of results.
                If not set, will use the index in the input list as ID.
            store_generation_traces: bool flag indicating if all intermediate steps
                and decisions should be recorded; for example for visualisations
                and debugging purposes.
            max_num_steps: maximal number of decoding steps (each creating at
                least one atom or bond) to perform.
            beam_size: number of rays in the beam to use.

        Returns:
            List of `MoLeRDecoderState` objects, which contain the final molecule,
            as well as information about the generation process.
            Note: The outputs are usually not in the same order as the inputs, and
            it is of size up to `beam_size * len(graph_representations)`.
            It is the responsibility of the caller to collect results as necessary.
        """
        if initial_molecules is None:
            initial_molecules = [None] * len(graph_representations)

        # Replace `None` in initial_molecules with empty molecules.
        initial_molecules = [
            Chem.Mol() if initMol is None else initMol for initMol in initial_molecules
        ]

        if len(graph_representations) != len(initial_molecules):
            raise ValueError(
                f"Number of graph representations ({len(graph_representations)})"
                f" and initial molecules ({len(initial_molecules)}) needs to match!"
            )
        if mol_ids is None:
            mol_ids = range(len(graph_representations))

        decoder_states: List[MoLeRDecoderState] = []
        for graph_repr, init_mol, mol_id in zip(graph_representations, initial_molecules, mol_ids):
            num_free_bond_slots = [0] * len(init_mol.GetAtoms())

            atom_id_pairs_to_disconnect: List[Tuple[int, int]] = []
            atom_ids_to_keep: List[int] = []

            for atom in init_mol.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    # Atomic number 0 means a placeholder atom that signifies an attachment point.
                    bonds = atom.GetBonds()

                    if len(bonds) > 1:
                        scaffold = Chem.MolToSmiles(init_mol)
                        raise ValueError(
                            f"Scaffold {scaffold} contains a [*] atom with at least two bonds."
                        )

                    if not bonds:
                        # This is a very odd case: either the scaffold we got is disconnected, or
                        # it consists of just a single * atom.
                        scaffold = Chem.MolToSmiles(init_mol)
                        raise ValueError(f"Scaffold {scaffold} contains a [*] atom with no bonds.")

                    [bond] = bonds
                    begin_idx = bond.GetBeginAtomIdx()
                    end_idx = bond.GetEndAtomIdx()

                    neighbour_idx = begin_idx if begin_idx != atom.GetIdx() else end_idx
                    num_free_bond_slots[neighbour_idx] += 1

                    atom_id_pairs_to_disconnect.append((atom.GetIdx(), neighbour_idx))
                else:
                    atom_ids_to_keep.append(atom.GetIdx())

            init_mol_original = init_mol
            if not atom_id_pairs_to_disconnect:
                # No explicit attachment points, so assume we can connect anywhere.
                num_free_bond_slots = None
            else:
                num_free_bond_slots = [num_free_bond_slots[idx] for idx in atom_ids_to_keep]
                init_mol = Chem.RWMol(init_mol)

                # Save the atom list to be able to extract neighbour atoms by their original id.
                original_atom_list = list(init_mol.GetAtoms())

                # Remove atoms starting from largest index, so that we don't have to account for
                # indices of atoms to remove shifting due to other removals.
                for atom_idx, neighbour_idx in reversed(atom_id_pairs_to_disconnect):
                    init_mol.RemoveAtom(atom_idx)

                    neighbour_atom = original_atom_list[neighbour_idx]
                    neighbour_atom.SetNumExplicitHs(neighbour_atom.GetNumExplicitHs() + 1)

                # Determine how the scaffold atoms will get reordered when we canonicalize it, so we can
                # permute `num_free_bond_slots` appropriately.
                canonical_ordering = compute_canonical_atom_order(init_mol)
                num_free_bond_slots = [num_free_bond_slots[idx] for idx in canonical_ordering]

            # Now canonicalize, which renumbers all the atoms, but we've applied the same
            # renumbering to `num_free_bond_slots` earlier.
            init_mol = Chem.MolFromSmiles(Chem.MolToSmiles(init_mol))

            if init_mol is None:
                scaffold = Chem.MolToSmiles(init_mol_original)
                raise ValueError(f"Scaffold {scaffold} could not be processed")

            # Clear aromatic flags in the scaffold, since partial graphs during training never have
            # them set (however we _do_ run `AtomIsAromaticFeatureExtractor`, it just always returns
            # 0 for partial graphs during training).
            # TODO(kmaziarz): Consider fixing this.
            Chem.Kekulize(init_mol, clearAromaticFlags=True)

            init_atom_types = []
            # TODO(kmaziarz): We need to be more careful in how the initial molecule looks like, to
            # make sure that `init_mol`s have correct atom features (e.g. charges).
            for atom in init_mol.GetAtoms():
                init_atom_types.append(get_atom_symbol(atom))
            adjacency_lists: List[List[Tuple[int, int]]] = [[] for _ in range(len(BOND_DICT))]
            for bond in init_mol.GetBonds():
                bond_type_idx = BOND_DICT[str(bond.GetBondType())]
                adjacency_lists[bond_type_idx].append(
                    (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                )
                adjacency_lists[bond_type_idx].append(
                    (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())
                )

            if self.uses_motifs:
                init_mol_motifs = find_motifs_from_vocabulary(
                    molecule=init_mol, motif_vocabulary=self._motif_vocabulary
                )
            else:
                init_mol_motifs = []

            decoder_states.append(
                MoLeRDecoderState(
                    molecule_representation=graph_repr,
                    molecule_id=mol_id,
                    molecule=init_mol,
                    atom_types=init_atom_types,
                    adjacency_lists=adjacency_lists,
                    visited_atoms=[atom.GetIdx() for atom in init_mol.GetAtoms()],
                    atoms_to_visit=[],
                    focus_atom=None,
                    # Pseudo-randomly pick last atom from input:
                    prior_focus_atom=len(init_atom_types) - 1,
                    generation_steps=[] if store_generation_traces else None,
                    motifs=init_mol_motifs,
                    num_free_bond_slots=num_free_bond_slots,
                )
            )

        decoder_states_empty: List[MoLeRDecoderState] = []
        decoder_states_non_empty: List[MoLeRDecoderState] = []

        for decoder_state in decoder_states:
            if decoder_state.molecule.GetNumAtoms() == 0:
                decoder_states_empty.append(decoder_state)
            else:
                decoder_states_non_empty.append(decoder_state)

        # Step 0: Pick first node types for states that do not have an initial molecule.
        first_node_pick_results = self._decoder_pick_first_atom_types(
            decoder_states=decoder_states_empty, num_samples=beam_size, sampling_mode=sampling_mode
        )

        # print("I: Picked first node types:", [picks for picks, _ in first_node_pick_results])

        decoder_states = decoder_states_non_empty

        for decoder_state, (first_node_type_picks, first_node_type_logprobs) in zip(
            decoder_states_empty, first_node_pick_results
        ):
            for first_node_type_pick, first_node_type_logprob in first_node_type_picks:
                # Set up generation trace storing variables, populating if needed.
                atom_choice_info = None
                if store_generation_traces:
                    atom_choice_info = MoleculeGenerationAtomChoiceInfo(
                        node_idx=0,
                        true_type_idx=None,
                        type_idx_to_prob=np.exp(first_node_type_logprobs),
                    )

                new_decoder_state, added_motif = self._add_atom_or_motif(
                    decoder_state,
                    first_node_type_pick,
                    logprob=first_node_type_logprob,
                    choice_info=atom_choice_info,
                )

                last_atom_id = new_decoder_state.molecule.GetNumAtoms() - 1

                if added_motif:
                    # To make all asserts happy, pretend we chose an attachment point.
                    new_decoder_state._focus_atom = last_atom_id

                # Mark all initial nodes as visited.
                new_decoder_state = MoLeRDecoderState.new_with_focus_marked_as_visited(
                    old_state=new_decoder_state, focus_node_finished_logprob=0.0
                )

                # Set the prior focus atom similarly to the start-from-scaffold case.
                new_decoder_state._prior_focus_atom = last_atom_id

                decoder_states.append(new_decoder_state)

        num_steps = 0
        while num_steps < max_num_steps:
            # This will hold the results after this decoding step, grouped by input mol id:
            new_decoder_states: List[MoLeRDecoderState] = []
            num_steps += 1
            # Step 1: Split decoder states into subsets, dependent on what they need next:
            require_atom_states, require_bond_states, require_attachment_point_states = [], [], []
            for decoder_state in decoder_states:
                # No focus atom => needs a new atom
                if decoder_state.focus_atom is None:
                    require_atom_states.append(decoder_state)
                # Focus atom has invalid index => decoding finished, just push forward unchanged:
                elif decoder_state.focus_atom < 0:
                    new_decoder_states.append(decoder_state)
                else:
                    require_bond_states.append(decoder_state)

            # Check if we are done:
            if (len(require_atom_states) + len(require_bond_states)) == 0:
                # print("I: Decoding finished")
                break

            # Step 2: For states that require a new atom, try to pick one:
            node_pick_results = self._decoder_pick_new_atom_types(
                decoder_states=require_atom_states,
                num_samples=beam_size,
                sampling_mode=sampling_mode,
            )

            for decoder_state, (node_type_picks, node_type_logprobs) in zip(
                require_atom_states, node_pick_results
            ):
                for node_type_pick, node_type_logprob in node_type_picks:
                    # Set up generation trace storing variables, populating if needed.
                    atom_choice_info = None
                    if store_generation_traces:
                        atom_choice_info = MoleculeGenerationAtomChoiceInfo(
                            node_idx=decoder_state.prior_focus_atom + 1,
                            true_type_idx=None,
                            type_idx_to_prob=np.exp(node_type_logprobs),
                        )

                    # If the decoder says we need no new atoms anymore, we are finished. Otherwise,
                    # start adding more bonds:
                    if node_type_pick is None:
                        # print(I {decoder_state.molecule_id} {decoder_state.logprob:12f}: Finished decoding - p={node_type_logprob:5f}")
                        new_decoder_states.append(
                            MoLeRDecoderState.new_for_finished_decoding(
                                old_state=decoder_state,
                                finish_logprob=node_type_logprob,
                                atom_choice_info=atom_choice_info,
                            )
                        )
                    else:
                        new_decoder_state, added_motif = self._add_atom_or_motif(
                            decoder_state,
                            node_type_pick,
                            logprob=node_type_logprob,
                            choice_info=atom_choice_info,
                        )

                        if added_motif:
                            require_attachment_point_states.append(new_decoder_state)
                        else:
                            require_bond_states.append(new_decoder_state)

            if self.uses_motifs:
                # Step 2': For states that require picking an attachment point, pick one:
                require_attachment_point_states = restrict_to_beam_size_per_mol(
                    require_attachment_point_states, beam_size
                )
                (
                    attachment_pick_results,
                    attachment_pick_logits,
                ) = self._decoder_pick_attachment_points(
                    decoder_states=require_attachment_point_states, sampling_mode=sampling_mode
                )

                for decoder_state, attachment_point_picks, attachment_point_logits in zip(
                    require_attachment_point_states,
                    attachment_pick_results,
                    attachment_pick_logits,
                ):
                    for attachment_point_pick, attachment_point_logprob in attachment_point_picks:
                        attachment_point_choice_info = None
                        if store_generation_traces:
                            attachment_point_choice_info = MoleculeGenerationAttachmentPointChoiceInfo(
                                partial_molecule_adjacency_lists=decoder_state.adjacency_lists,
                                motif_nodes=decoder_state.atoms_to_mark_as_visited,
                                candidate_attachment_points=decoder_state.candidate_attachment_points,
                                candidate_idx_to_prob=tf.nn.softmax(attachment_point_logits),
                                correct_attachment_point_idx=None,
                            )

                        # print(I {decoder_state.molecule_id} {decoder_state.logprob:12f}: Picked attachment point {attachment_point_pick} - p={attachment_point_logprob:5f}")
                        require_bond_states.append(
                            MoLeRDecoderState.new_with_focus_on_attachment_point(
                                decoder_state,
                                attachment_point_pick,
                                focus_atom_logprob=attachment_point_logprob,
                                attachment_point_choice_info=attachment_point_choice_info,
                            )
                        )
            else:
                assert not require_attachment_point_states

            # Step 3: Pick fresh bonds and populate the next round of decoding steps:
            require_bond_states = restrict_to_beam_size_per_mol(require_bond_states, beam_size)
            bond_pick_results = self._decoder_pick_new_bond_types(
                decoder_states=require_bond_states,
                store_generation_traces=store_generation_traces,
                sampling_mode=sampling_mode,
            )
            for decoder_state, (bond_picks, edge_choice_info) in zip(
                require_bond_states, bond_pick_results
            ):
                if len(bond_picks) == 0:
                    # There were no valid options for this bonds, so we treat this as if
                    # predicting no more bonds with probability 1.0:
                    # print(I {decoder_state.molecule_id} {decoder_state.logprob:12f}: No more allowed bonds to node {decoder_state.focus_atom}")
                    new_decoder_states.append(
                        MoLeRDecoderState.new_with_focus_marked_as_visited(
                            decoder_state,
                            focus_node_finished_logprob=0,
                            edge_choice_info=edge_choice_info,
                        )
                    )
                    continue

                for bond_pick, bond_pick_logprob in bond_picks:
                    # If the decoder says we need no more bonds for the current focus node,
                    # we mark this and put the decoder state back for the next expansion round:
                    if bond_pick is None:
                        # print(I {decoder_state.molecule_id} {decoder_state.logprob:12f}: Finished connecting bonds to node {decoder_state.focus_atom} - p={bond_pick_logprob:5f}")
                        new_decoder_states.append(
                            MoLeRDecoderState.new_with_focus_marked_as_visited(
                                decoder_state,
                                focus_node_finished_logprob=bond_pick_logprob,
                                edge_choice_info=edge_choice_info,
                            )
                        )
                    else:
                        (picked_bond_target, picked_bond_type) = bond_pick

                        # print(I {decoder_state.molecule_id} {decoder_state.logprob:12f}: Adding {decoder_state.focus_atom}-{picked_bond_type}->{picked_bond_target} - p={bond_pick_logprob:5f}")
                        new_decoder_states.append(
                            MoLeRDecoderState.new_with_added_bond(
                                old_state=decoder_state,
                                target_atom_idx=int(
                                    picked_bond_target
                                ),  # Go from np.int32 to pyInt
                                bond_type_idx=picked_bond_type,
                                bond_logprob=bond_pick_logprob,
                                edge_choice_info=edge_choice_info,
                            )
                        )

            # Everything is done, restrict to the beam width, and go back to the loop start:
            decoder_states = restrict_to_beam_size_per_mol(new_decoder_states, beam_size)

        return decoder_states

    def _decoder_pick_new_atom_types(
        self,
        *,
        decoder_states: List[MoLeRDecoderState],
        sampling_mode: DecoderSamplingMode,
        num_samples: int = 1,
    ) -> Iterable[Tuple[List[Tuple[Optional[str], float]], np.ndarray]]:
        """
        Query the model to pick a new atom to add for each of a list of decoder states.

        Args:
            decoder_states: MoLeRDecoderState objects representing partial
                results of the decoder that are to be extended by an additional atom.
            sampling_mode: Determines how to obtain num_samples. GREEDY takes the most
                likely values, whereas SAMPLING samples according to the predicted
                probabilities.
            num_samples: Number of samples to return per input decoder state (non-1 values
                are useful for beam search).

        Returns:
            An iterable of the same length as the input list `decoder_states`, such that
            the i-th entry of the result is the result for the i-th input.
            The result is either the string representation of the picked atom (e.g.,
            "C" or "N+") or `None`, if the model decided that no further atoms should be
            added.
        """
        if len(decoder_states) == 0:
            return []

        # We first need to create a minibatch of all of our partial graphs here:
        def init_atom_choice_batch(batch: Dict[str, Any]):
            batch["prior_focus_atoms"] = []

        def add_state_to_atom_choice_batch(batch: Dict[str, Any], decoder_state: MoLeRDecoderState):
            batch["prior_focus_atoms"].append(
                decoder_state.prior_focus_atom + batch["nodes_in_batch"]
            )

        batch_generator = self._batch_decoder_states(
            decoder_states=decoder_states,
            init_batch_callback=init_atom_choice_batch,
            add_state_to_batch_callback=add_state_to_atom_choice_batch,
        )
        atom_type_pick_generator = (
            self._pick_new_atom_types_for_batch(batch, num_samples, sampling_mode)
            for batch, _ in batch_generator
        )
        return itertools.chain.from_iterable(atom_type_pick_generator)

    def _pick_new_atom_types_for_batch(
        self, batch: Dict[str, Any], num_samples: int, sampling_mode: DecoderSamplingMode
    ):
        graph_representations, _ = self.calculate_node_and_graph_representations(
            node_features=batch["node_features"],
            node_categorical_features=batch["node_categorical_features"],
            adjacency_lists=batch["adjacency_lists"],
            num_graphs_in_batch=batch["graphs_in_batch"],
            node_to_graph_map=batch["node_to_graph_map"],
            # Note: This whole prior_focus_atom is a bit of a hack. During training, we use the
            # same graph for predict-no-more-bonds and predict-next-atom-type. Hence, during
            # training, we always have at least one in-focus node per graph, and not
            # matching that would be confusing to the model. Hence, we simulate this behaviour:
            graph_to_focus_node_map=batch["prior_focus_atoms"],
            candidate_attachment_points=np.zeros(shape=(0,)),
            training=False,
        )

        node_type_logits = self.pick_node_type(
            input_molecule_representations=batch["molecule_representations"],
            graph_representations=graph_representations,
            graphs_requiring_node_choices=np.arange(0, batch["graphs_in_batch"]),
            training=False,
        )  # Shape [G, NT + 1]

        # Remove the first column, corresponding to UNK, which we never want to produce, but add it
        # back later so that the type lookup indices work out:
        atom_type_logprobs = tf.nn.log_softmax(
            node_type_logits[:, 1:], axis=1
        ).numpy()  # Shape [G, NT]
        atom_type_pick_results: List[Tuple[List[Tuple[Optional[str], float]], np.ndarray]] = []
        # Iterate over each of the rows independently, sampling for each input state:
        for state_atom_type_logprobs in atom_type_logprobs:
            picked_atom_type_indices = sample_indices_from_logprobs(
                num_samples, sampling_mode, state_atom_type_logprobs
            )

            this_state_results: List[Tuple[Optional[str], float]] = []
            for picked_atom_type_idx in picked_atom_type_indices:
                pick_logprob: float = state_atom_type_logprobs[picked_atom_type_idx]
                picked_atom_type_idx += 1  # Revert the stripping out of the UNK (index 0) type
                # This is the case in which we picked the "no further nodes" virtual node type:
                if picked_atom_type_idx >= self._num_node_types:
                    this_state_results.append((None, pick_logprob))
                else:
                    picked_atom_type = self._index_to_node_type_map[picked_atom_type_idx]
                    this_state_results.append((picked_atom_type, pick_logprob))
            atom_type_pick_results.append((this_state_results, state_atom_type_logprobs))
        return atom_type_pick_results

    def _decoder_pick_first_atom_types(
        self,
        *,
        decoder_states: List[MoLeRDecoderState],
        sampling_mode: DecoderSamplingMode,
        num_samples: int = 1,
    ) -> Iterable[Tuple[List[Tuple[Optional[str], float]], np.ndarray]]:
        """
        Query the model to pick the first atom to add for each of a list of decoder states.

        Args:
            decoder_states: MoLeRDecoderState objects representing partial
                results of the decoder that are to be extended by an additional atom.
            sampling_mode: Determines how to obtain num_samples. GREEDY takes the most
                likely values, whereas SAMPLING samples according to the predicted
                probabilities.
            num_samples: Number of samples to return per input decoder state (non-1 values
                are useful for beam search).

        Returns:
            A list of the same length as the input list `decoder_states`; the i-th entry
            is the result for the i-th input.
            A single result contains a list of `num_samples` first node type choices, and
            an array containing all the first node type logprobs.
        """
        if len(decoder_states) == 0:
            return []

        if not self.has_first_node_type_selector:
            # Models without the first node type selector always start decoding from carbon.
            def generate_result_that_picks_carbon():
                first_node_type_picks = [("C", 0.0) for _ in range(num_samples)]

                # Probability of selecting anything other than carbon is 0.
                first_node_type_logprobs = np.full(self._num_node_types, -BIG_NUMBER)
                first_node_type_logprobs[self._carbon_type_idx] = 0.0

                return first_node_type_picks, first_node_type_logprobs

            return [generate_result_that_picks_carbon() for _ in range(len(decoder_states))]

        # We only need the molecule representations.
        molecule_representations = np.stack(
            [state.molecule_representation for state in decoder_states]
        )

        first_node_type_logits = self.pick_first_node_type(
            input_molecule_representations=molecule_representations, training=False
        )  # Shape [G, NT + 1]

        first_atom_type_logprobs = tf.nn.log_softmax(
            first_node_type_logits[:, 1:], axis=1
        ).numpy()  # Shape [G, NT]

        first_atom_type_pick_results: List[
            Tuple[List[Tuple[Optional[str], float]], np.ndarray]
        ] = []

        # Iterate over each of the rows independently, sampling for each input state:
        for state_first_atom_type_logprobs in first_atom_type_logprobs:
            picked_atom_type_indices = sample_indices_from_logprobs(
                num_samples, sampling_mode, state_first_atom_type_logprobs
            )

            this_state_results: List[Tuple[Optional[str], float]] = []

            for picked_atom_type_idx in picked_atom_type_indices:
                pick_logprob: float = state_first_atom_type_logprobs[picked_atom_type_idx]
                picked_atom_type_idx += 1  # Revert the stripping out of the UNK (index 0) type

                this_state_results.append(
                    (self._index_to_node_type_map[picked_atom_type_idx], pick_logprob)
                )

            first_atom_type_pick_results.append(
                (this_state_results, state_first_atom_type_logprobs)
            )

        return first_atom_type_pick_results

    def _decoder_pick_new_bond_types(
        self,
        *,
        decoder_states: List[MoLeRDecoderState],
        sampling_mode: DecoderSamplingMode,
        store_generation_traces: bool = False,
        num_samples: int = 1,
    ) -> Iterable[
        Tuple[
            List[Tuple[Optional[Tuple[int, int]], float]],
            Optional[MoleculeGenerationEdgeChoiceInfo],
        ]
    ]:
        """
        Query the model to pick a new bond to add for each of a list of decoder states.

        Args:
            decoder_states: MoLeRDecoderState objects representing partial
                results of the decoder that are to be extended by an additional bond.
            store_generation_traces: Bool denoting whether `MoleculeGenerationEdgeChoiceInfo`
                should be calculated and returned with the function.
            sampling_mode: Determines how to obtain num_samples. GREEDY takes the most
                likely values, whereas SAMPLING samples according to the predicted
                probabilities.
            num_samples: Number of samples to return per input decoder state (non-1 values
                are useful for beam search).

        Returns:
            An iterable of the same length as the input list `decoder_states`, such that
            the i-th entry of the result is the result for the i-th input.
            The result is either a pair `(idx, typ_idx)` of integers, where `idx` is the
            index of the atom the current focus node should be connected to and `type_idx`
            indicates the type of bond to add, or `None`, if the model decided that no
            further bonds should be connected to the current atom in focus.
        """
        if len(decoder_states) == 0:
            return []

        # We first need to create a minibatch of all of our partial graphs here:
        def init_edge_batch(batch: Dict[str, Any]):
            batch["focus_atoms"] = []
            batch["candidate_edge_targets"] = []
            batch["candidate_edge_targets_offset"] = []  # needed to get original target node ID
            batch["candidate_edge_type_masks"] = []
            batch["candidate_edge_features"] = []
            batch["decoder_state_to_num_candidate_edges"] = []

        def add_state_to_edge_batch(batch: Dict[str, Any], decoder_state: MoLeRDecoderState):
            batch["focus_atoms"].append(decoder_state.focus_atom + batch["nodes_in_batch"])
            candidate_targets, candidate_bond_type_mask = decoder_state.get_bond_candidate_targets()
            num_edge_candidates = len(candidate_targets)
            batch["candidate_edge_targets"].append(candidate_targets + batch["nodes_in_batch"])
            batch["candidate_edge_targets_offset"].append(batch["nodes_in_batch"])
            batch["candidate_edge_type_masks"].append(candidate_bond_type_mask)
            batch["candidate_edge_features"].append(
                decoder_state.compute_bond_candidate_features(candidate_targets)
            )
            batch["decoder_state_to_num_candidate_edges"].append(num_edge_candidates)

        batch_generator = self._batch_decoder_states(
            decoder_states=decoder_states,
            init_batch_callback=init_edge_batch,
            add_state_to_batch_callback=add_state_to_edge_batch,
        )

        picked_edges_generator = (
            self._pick_edges_for_batch(b, d, num_samples, sampling_mode, store_generation_traces)
            for b, d in batch_generator
        )
        return itertools.chain.from_iterable(picked_edges_generator)

    def _pick_edges_for_batch(
        self,
        batch: Dict[str, Any],
        decoder_states: List[MoLeRDecoderState],
        num_samples: int,
        sampling_mode: DecoderSamplingMode,
        store_generation_traces: bool,
    ):
        graph_representations, node_representations = self.calculate_node_and_graph_representations(
            node_features=batch["node_features"],
            node_categorical_features=batch["node_categorical_features"],
            adjacency_lists=batch["adjacency_lists"],
            num_graphs_in_batch=batch["graphs_in_batch"],
            node_to_graph_map=batch["node_to_graph_map"],
            graph_to_focus_node_map=batch["focus_atoms"],
            candidate_attachment_points=np.zeros(shape=(0,)),
            training=False,
        )

        batch_candidate_edge_targets = np.concatenate(batch["candidate_edge_targets"], axis=0)
        batch_candidate_edge_type_masks = np.concatenate(batch["candidate_edge_type_masks"], axis=0)

        edge_candidate_logits, edge_type_logits = self.pick_edge(
            input_molecule_representations=batch["molecule_representations"],
            graph_representations=graph_representations,
            node_representations=node_representations,
            num_graphs_in_batch=batch["graphs_in_batch"],
            graph_to_focus_node_map=batch["focus_atoms"],
            node_to_graph_map=batch["node_to_graph_map"],
            candidate_edge_targets=batch_candidate_edge_targets,
            candidate_edge_features=np.concatenate(batch["candidate_edge_features"], axis=0).astype(
                np.float32
            ),
            training=False,
        )  # Shape [VE + PG, 1], [VE, ET]

        # We now need to unpack the results, which is a bit fiddly because the "no more edges"
        # logits are bunched together at the end for all input graphs...
        num_total_edge_candidates = sum(batch["decoder_state_to_num_candidate_edges"])
        edge_candidate_offset = 0
        picked_edges: List[
            Tuple[
                List[Tuple[Optional[Tuple[int, int]], float]],
                Optional[MoleculeGenerationEdgeChoiceInfo],
            ]
        ] = []
        for state_idx, (decoder_state, decoder_state_num_edge_candidates) in enumerate(
            zip(decoder_states, batch["decoder_state_to_num_candidate_edges"])
        ):
            # We had no valid candidates -> Easy out:
            if decoder_state_num_edge_candidates == 0:
                picked_edges.append(([], None))
                continue

            # Find the edge targets for this decoder state, in the original node index:
            edge_targets = batch_candidate_edge_targets[
                edge_candidate_offset : edge_candidate_offset + decoder_state_num_edge_candidates
            ]
            edge_targets_orig_idx = edge_targets - batch["candidate_edge_targets_offset"][state_idx]

            # Get logits for edge candidates for this decoder state:
            decoder_state_edge_candidate_logits = edge_candidate_logits[
                edge_candidate_offset : edge_candidate_offset + decoder_state_num_edge_candidates
            ]
            decoder_state_no_edge_logit = edge_candidate_logits[
                num_total_edge_candidates + state_idx
            ]

            decoder_state_edge_cand_logprobs = tf.nn.log_softmax(
                tf.concat(
                    [decoder_state_edge_candidate_logits, [decoder_state_no_edge_logit]], axis=0
                )
            )

            # Before we continue, generate the information for the trace visualisation:
            molecule_generation_edge_choice_info = None
            if store_generation_traces:
                # Set up the edge candidate info
                candidate_edge_type_logits = edge_type_logits[
                    edge_candidate_offset : edge_candidate_offset
                    + decoder_state_num_edge_candidates
                ]
                candidate_edge_type_mask = batch_candidate_edge_type_masks[
                    edge_candidate_offset : edge_candidate_offset
                    + decoder_state_num_edge_candidates
                ]
                masked_candidate_edge_type_logits = candidate_edge_type_logits - BIG_NUMBER * (
                    1 - candidate_edge_type_mask
                )

                # Loop over the edge candidates themselves.
                molecule_generation_edge_candidate_info = []
                for edge_idx, (target, score, logprob) in enumerate(
                    zip(
                        edge_targets_orig_idx,
                        decoder_state_edge_candidate_logits,
                        decoder_state_edge_cand_logprobs,
                    )
                ):
                    molecule_generation_edge_candidate_info.append(
                        MoleculeGenerationEdgeCandidateInfo(
                            target_node_idx=target,
                            score=score,
                            logprob=logprob,
                            correct=None,
                            type_idx_to_logprobs=tf.nn.log_softmax(
                                masked_candidate_edge_type_logits[edge_idx, :]
                            ),
                        )
                    )
                molecule_generation_edge_choice_info = MoleculeGenerationEdgeChoiceInfo(
                    focus_node_idx=decoder_state.focus_atom,
                    partial_molecule_adjacency_lists=decoder_state.adjacency_lists,
                    candidate_edge_infos=molecule_generation_edge_candidate_info,
                    no_edge_score=decoder_state_no_edge_logit,
                    no_edge_logprob=decoder_state_edge_cand_logprobs[-1],
                    no_edge_correct=None,
                )

            # Collect (sampling) results for this state:
            this_state_results: List[Tuple[Optional[Tuple[int, int]], float]] = []
            picked_edge_cand_indices = sample_indices_from_logprobs(
                num_samples, sampling_mode, decoder_state_edge_cand_logprobs
            )
            for picked_edge_cand_idx in picked_edge_cand_indices:
                picked_cand_logprob = decoder_state_edge_cand_logprobs[picked_edge_cand_idx]
                # Handle case of having no edge is better:
                if picked_edge_cand_idx == len(decoder_state_edge_cand_logprobs) - 1:
                    this_state_results.append((None, picked_cand_logprob))
                else:
                    # Otherwise, we need to find the target of that edge, in the original
                    # (unbatched) node index:
                    picked_edge_partner = edge_targets_orig_idx[picked_edge_cand_idx]

                    # Next, identify the edge type for this choice:
                    edge_type_mask = batch_candidate_edge_type_masks[
                        edge_candidate_offset + picked_edge_cand_idx
                    ]
                    cand_edge_type_logprobs = tf.nn.log_softmax(
                        edge_type_logits[edge_candidate_offset + picked_edge_cand_idx]
                        - BIG_NUMBER * (1 - edge_type_mask)
                    )
                    picked_edge_types = sample_indices_from_logprobs(
                        num_samples, sampling_mode, cand_edge_type_logprobs
                    )
                    for picked_edge_type in picked_edge_types:
                        picked_edge_logprob = (
                            picked_cand_logprob + cand_edge_type_logprobs[picked_edge_type]
                        )
                        this_state_results.append(
                            ((picked_edge_partner, picked_edge_type), picked_edge_logprob)
                        )
            picked_edges.append((this_state_results, molecule_generation_edge_choice_info))
            edge_candidate_offset += decoder_state_num_edge_candidates

        return picked_edges

    def _decoder_pick_attachment_points(
        self,
        *,
        decoder_states: List[MoLeRDecoderState],
        sampling_mode: DecoderSamplingMode,
        num_samples: int = 1,
    ) -> Tuple[List[List[Tuple[int, float]]], List[List[float]]]:
        """
        Query the model to pick a motif attachment point for each decoder state in a list.

        Args:
            decoder_states: MoLeRDecoderState objects representing partial
                results of the decoder that need attachment point selection.
            store_generation_traces: Bool denoting whether `MoleculeGenerationEdgeChoiceInfo`
                should be calculated and returned with the function.
            sampling_mode: Determines how to obtain num_samples. GREEDY takes the most
                likely values, whereas SAMPLING samples according to the predicted
                probabilities.
            num_samples: Number of samples to return per input decoder state (non-1 values
                are useful for beam search).

        Returns:
            Two lists of the same length as the input list `decoder_states`, such that the
            i-th entry of either list is the result for the i-th input.
            The first list contains integers, which denote the chosen attachment points. The second
            list contains lists of floats, which denote the logits for all candidates.
        """
        if len(decoder_states) == 0:
            return [], np.zeros(shape=(0,))

        # We first need to create a minibatch of all of our partial graphs here:
        def init_attachment_point_choice_batch(batch: Dict[str, Any]):
            batch["candidate_attachment_points"] = []

        def add_state_to_attachment_point_choice_batch(
            batch: Dict[str, Any], decoder_state: MoLeRDecoderState
        ):
            batch["candidate_attachment_points"].append(
                np.array(decoder_state.candidate_attachment_points) + batch["nodes_in_batch"]
            )

        attachment_point_pick_results = []
        logits_by_graph = []

        for batch, decoder_states_batch in self._batch_decoder_states(
            decoder_states=decoder_states,
            init_batch_callback=init_attachment_point_choice_batch,
            add_state_to_batch_callback=add_state_to_attachment_point_choice_batch,
        ):
            pick_results_for_batch, logits_for_batch = self._pick_attachment_points_for_batch(
                batch=batch,
                decoder_states=decoder_states_batch,
                num_samples=num_samples,
                sampling_mode=sampling_mode,
            )
            attachment_point_pick_results.extend(pick_results_for_batch)
            logits_by_graph.extend(logits_for_batch)

        return attachment_point_pick_results, logits_by_graph

    def _pick_attachment_points_for_batch(
        self,
        batch: Dict[str, Any],
        decoder_states: List[MoLeRDecoderState],
        num_samples: int,
        sampling_mode: DecoderSamplingMode,
    ):
        focus_atoms = np.array(
            [attachment_points[0] for attachment_points in batch["candidate_attachment_points"]]
        )
        candidate_attachment_points = np.concatenate(batch["candidate_attachment_points"], axis=0)

        graph_representations, node_representations = self.calculate_node_and_graph_representations(
            node_features=batch["node_features"],
            node_categorical_features=batch["node_categorical_features"],
            adjacency_lists=batch["adjacency_lists"],
            num_graphs_in_batch=batch["graphs_in_batch"],
            node_to_graph_map=batch["node_to_graph_map"],
            # Here we choose an arbitrary attachment point as a focus atom; this does not matter
            # since later all candidate attachment points are marked with the in-focus bit.
            graph_to_focus_node_map=focus_atoms,
            candidate_attachment_points=candidate_attachment_points,
            training=False,
        )

        attachment_point_selection_logits = self.pick_attachment_point(
            input_molecule_representations=batch["molecule_representations"],
            graph_representations=graph_representations,
            node_representations=node_representations,
            node_to_graph_map=batch["node_to_graph_map"],
            candidate_attachment_points=candidate_attachment_points,
            training=False,
        )  # Shape: [CA]

        attachment_point_to_graph_map = tf.gather(
            batch["node_to_graph_map"], candidate_attachment_points
        )

        # TODO(krmaziar): Consider tensorizing the code below. For that, we need some equivalent of
        # `unsorted_segment_argmax`.
        logits_by_graph: List[List[float]] = [[] for _ in range(len(decoder_states))]

        for logit, graph_id in zip(
            attachment_point_selection_logits, attachment_point_to_graph_map
        ):
            logits_by_graph[graph_id].append(logit)

        attachment_point_pick_results: List[List[Tuple[int, float]]] = []
        for old_decoder_state, attachment_point_logits in zip(decoder_states, logits_by_graph):
            attachment_point_logprobs = tf.nn.log_softmax(attachment_point_logits).numpy()
            picked_att_point_indices = sample_indices_from_logprobs(
                num_samples, sampling_mode, attachment_point_logprobs
            )

            this_state_results = []
            for attachment_point_pick_idx in picked_att_point_indices:
                attachment_point_pick = old_decoder_state.candidate_attachment_points[
                    attachment_point_pick_idx
                ]
                attachment_point_logprob = attachment_point_logprobs[attachment_point_pick_idx]

                this_state_results.append((attachment_point_pick, attachment_point_logprob))
            attachment_point_pick_results.append(this_state_results)

        return attachment_point_pick_results, logits_by_graph


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
