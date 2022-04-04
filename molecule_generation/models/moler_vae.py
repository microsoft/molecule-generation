"""MoLeR Variational Autoencoder model."""
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, NamedTuple, Union, Iterable

import numpy as np
import tensorflow as tf
from dpu_utils.tf2utils import MLP
from tf2_gnn import GraphTaskModel
from tf2_gnn.layers import NodesToGraphRepresentationInput, WeightedSumGraphRepresentation

from molecule_generation.dataset.trace_dataset import TraceDataset
from molecule_generation.layers.moler_decoder import (
    MoLeRDecoder,
    MoLeRDecoderInput,
    MoLeRDecoderMetrics,
)
from molecule_generation.utils.training_utils import get_class_balancing_weights
from molecule_generation.utils.property_models import (
    PropertyTaskType,
    PropertyPredictionLayer,
    MLPBinaryClassifierLayer,
    MLPRegressionLayer,
)
from molecule_generation.utils.epoch_metrics_logger import EpochMetricsLogger


@dataclass
class MoLeRVaeOutput:
    graph_representation_mean: tf.Tensor
    graph_representation_log_variance: tf.Tensor
    node_type_logits: tf.Tensor
    first_node_type_logits: tf.Tensor
    edge_candidate_logits: tf.Tensor
    edge_type_logits: tf.Tensor
    attachment_point_selection_logits: Optional[tf.Tensor]
    predicted_properties: Dict[str, tf.Tensor]


@dataclass
class MoLeRMetrics(MoLeRDecoderMetrics):
    loss: tf.Tensor
    kl_divergence: tf.Tensor


class PropertyPredictionMetrics(NamedTuple):
    loss: tf.Tensor
    property_to_metrics: Dict[str, Any]


class MoLeRVae(GraphTaskModel):
    """[Mo]lecule [Le]vel [R]representation-based VAE model for handling molecular graphs.

    The components of our model work together as follows:
        Molecule -> Encoder -> NodeToGraphRepresentation -> Sampling -> MoLeRDecoder -> Molecule

    Molecules are built both using single atoms and larger chunks (motifs).
    The vocabulary of motifs has to be precomputed during preprocessing.
    """

    __decoder_prefix = "decoder_"

    @classmethod
    def decoder_prefix(cls) -> str:
        return cls.__decoder_prefix

    @classmethod
    def get_default_hyperparameters(cls, mp_style: Optional[str] = None) -> Dict[str, Any]:
        base_hypers = super().get_default_hyperparameters(mp_style)

        # Add our own hyperparameters:
        base_hypers.update(
            {
                # Encoder GNN hyperparameters:
                "gnn_hidden_dim": 64,
                "gnn_num_layers": 12,
                "gnn_message_activation_function": "leaky_relu",
                "gnn_layer_input_dropout_rate": 0.0,
                "gnn_global_exchange_dropout_rate": 0.0,
                "gnn_use_inter_layer_layernorm": True,
                "use_intermediate_gnn_results": True,  # Use all intermediate results of the encoder GNN
                "categorical_features_embedding_dim": 64,
                # Encoder -> Graph-level representation hyperparameters:
                "latent_repr_size": 512,
                "latent_repr_mlp_layers": [512, 512],
                "latent_repr_num_heads": 32,
                "latent_repr_dropout_rate": 0.0,
                "latent_sample_strategy": "per_graph",  # one of passthrough, per_graph, or per_partial_graph
                # Property prediction hyperparameters:
                "property_predictor_mlp_layers": [64, 32],
                "property_predictor_mlp_dropout_rate": 0.0,
                # Relative contributions of the different next step prediction losses:
                "node_classification_loss_weight": 1.0,
                "first_node_classification_loss_weight": 0.07,
                "edge_selection_loss_weight": 1.0,
                "edge_type_loss_weight": 1.0,
                "attachment_point_selection_weight": 1.0,
                # Relative contribution of the KL loss term. This was tuned for a motif
                # vocabulary of 128, may need to be adjusted for larger vocabularies
                # (empirically increments of 0.005 every time vocabulary size doubles):
                "kl_divergence_weight": 0.02,
                "kl_divergence_annealing_beta": 0.999,  # 10% weight after 100 steps, 63% after 1000, 99% after 5000
                # Training hyperparameters:
                "learning_rate": 0.001,
                "gradient_clip_value": 0.5,
                "num_train_steps_between_valid": 0,  # Default value of 0 means that whole dataset will be run through.
                # Not really hyperparameters, but logging parameters:
                "logged_loss_smoothing_window_size": 100,  # Number of training batches to include in the reported moving average training loss.
            }
        )

        # Add hyperparameters for the GNN decoder:
        decoder_hypers = {
            cls.decoder_prefix() + k: v
            for k, v in MoLeRDecoder.get_default_params(mp_style).items()
        }
        base_hypers.update(decoder_hypers)

        return base_hypers

    def __init__(self, params: Dict[str, Any], dataset: TraceDataset, **kwargs):
        super().__init__(params, dataset, **kwargs)

        # Shortcuts to commonly used hyperparameters:
        self._latent_repr_dim = params["latent_repr_size"]
        self._latent_sample_strategy = params["latent_sample_strategy"]

        # Keep track of the training step as a TF variable
        self._train_step_counter = tf.Variable(
            initial_value=0, trainable=False, dtype=tf.int32, name="training_step"
        )

        # Get some information out from the dataset:
        next_node_type_distribution = dataset.metadata.get("train_next_node_type_distribution")

        # ===== Prepare sub-layers, which will be actually created in .build().
        # Layer from per-node encoder GNN results to per-graph representation:
        assert (
            self.latent_dim % 2 == 0
        ), "Latent dimension need to be divisble by two for technical implementation reasons."
        self._weighted_avg_of_nodes_to_graph_repr = WeightedSumGraphRepresentation(
            graph_representation_size=self.latent_dim // 2,
            num_heads=self._params["latent_repr_num_heads"] // 2,
            weighting_fun="softmax",
            scoring_mlp_layers=[l // 2 for l in self._params["latent_repr_mlp_layers"]],
            scoring_mlp_dropout_rate=self._params["latent_repr_dropout_rate"],
            transformation_mlp_layers=[l // 2 for l in self._params["latent_repr_mlp_layers"]],
            transformation_mlp_dropout_rate=self._params["latent_repr_dropout_rate"],
        )
        self._weighted_sum_of_nodes_to_graph_repr = WeightedSumGraphRepresentation(
            graph_representation_size=self.latent_dim // 2,
            num_heads=self._params["latent_repr_num_heads"] // 2,
            weighting_fun="sigmoid",
            scoring_mlp_layers=[l // 2 for l in self._params["latent_repr_mlp_layers"]],
            scoring_mlp_dropout_rate=self._params["latent_repr_dropout_rate"],
            transformation_mlp_layers=[l // 2 for l in self._params["latent_repr_mlp_layers"]],
            transformation_mlp_dropout_rate=self._params["latent_repr_dropout_rate"],
            transformation_mlp_result_upper_bound=5,
        )

        # Layer from per-graph representation to samples to be used in decoder:
        self._mean_and_var_mlp: MLP = MLP(
            out_size=self.latent_dim * 2,
            hidden_layers=0,
            use_biases=False,
            name="MeanAndVarMLP",
        )

        # Layers from per-graph representation to properties of these graphs:
        self._property_predictors: Dict[str, PropertyPredictionLayer] = {}
        self._property_predictor_params: Dict[str, Any] = {}
        property_predictor_default_hypers = {
            "mlp_layers": self._params["property_predictor_mlp_layers"],
            "dropout_rate": self._params["property_predictor_mlp_dropout_rate"],
        }
        for prop_name, prop_params in dataset.params["graph_properties"].items():
            prop_predictor_hypers = dict(property_predictor_default_hypers)
            prop_predictor_hypers.update(prop_params)
            self._property_predictor_params[prop_name] = prop_predictor_hypers
            prop_type = PropertyTaskType[prop_params["type"].upper()]

            if prop_type == PropertyTaskType.BINARY_CLASSIFICATION:
                self._property_predictors[prop_name] = MLPBinaryClassifierLayer(
                    mlp_layer_sizes=prop_predictor_hypers["mlp_layers"],
                    dropout_rate=prop_predictor_hypers["dropout_rate"],
                )
            elif prop_type == PropertyTaskType.REGRESSION:
                prop_stddev = dataset.metadata.get(f"{prop_name}_stddev")
                if not (prop_params.get("normalise_loss", True)):
                    prop_stddev = None
                self._property_predictors[prop_name] = MLPRegressionLayer(
                    mlp_layer_sizes=prop_predictor_hypers["mlp_layers"],
                    dropout_rate=prop_predictor_hypers["dropout_rate"],
                    property_stddev=prop_stddev,
                )
            else:
                raise ValueError(f"Unknown property type {prop_type}")

        class_weight_factor = self._params.get("node_type_predictor_class_loss_weight_factor", 0.0)

        if not (0.0 <= class_weight_factor <= 1.0):
            raise ValueError(
                f"Node class loss weight node_classifier_class_loss_weight_factor must be in [0,1], but is {class_weight_factor}!"
            )

        if class_weight_factor > 0:
            atom_type_nums = [
                next_node_type_distribution[dataset.node_type_index_to_string[type_idx]]
                for type_idx in range(dataset.num_node_types)
            ]
            atom_type_nums.append(next_node_type_distribution["None"])

            class_weights = get_class_balancing_weights(
                class_counts=atom_type_nums, class_weight_factor=class_weight_factor
            )
        else:
            class_weights = None

        motif_vocabulary = dataset.metadata.get("motif_vocabulary")
        self._uses_motifs = motif_vocabulary is not None

        self._node_categorical_num_classes = dataset.node_categorical_num_classes

        if self.uses_categorical_features:
            if "categorical_features_embedding_dim" in self._params:
                self._node_categorical_features_embedding = None
            else:
                # Older models use one hot vectors instead of dense embeddings, simulate that here.
                self._params[
                    "categorical_features_embedding_dim"
                ] = self._node_categorical_num_classes
                self._node_categorical_features_embedding = np.eye(
                    self._node_categorical_num_classes, dtype=np.float32
                )

        # Finally, the decoder layer, which does all kinds of important things:
        decoder_prefix = self.decoder_prefix()
        n = len(decoder_prefix)
        decoder_hypers = {k[n:]: v for k, v in self._params.items() if k.startswith(decoder_prefix)}
        self._decoder_layer = MoLeRDecoder(
            decoder_hypers,
            name="MoLeRDecoder",
            atom_featurisers=dataset.metadata.get("feature_extractors", []),
            index_to_node_type_map=dataset.node_type_index_to_string,
            node_type_loss_weights=class_weights,
            motif_vocabulary=motif_vocabulary,
            node_categorical_num_classes=self._node_categorical_num_classes,
        )

        # Moving average variables, will be filled later:
        self._logged_loss_smoothing_window_size = self._params["logged_loss_smoothing_window_size"]

        # Deal with Tensorboard's global state:
        tf.summary.experimental.set_step(0)

    @property
    def latent_dim(self):
        return self._latent_repr_dim

    @property
    def decoder(self):
        return self._decoder_layer

    @property
    def uses_motifs(self) -> bool:
        return self._uses_motifs

    @property
    def uses_categorical_features(self) -> bool:
        return self._node_categorical_num_classes is not None

    def build(self, input_shapes: Dict[str, Any]):
        # Compute some sizes and shapes we'll re-use a few times:
        final_node_representation_dim = self._params["gnn_hidden_dim"]
        if self._params["use_intermediate_gnn_results"]:
            # In this case, we have one initial representation + results for all layers:
            final_node_representation_dim *= 1 + self._params["gnn_num_layers"]
        latent_graph_representation_shape = tf.TensorShape((None, self.latent_dim))

        # Build the individual layers, which we've initialised in __init__():
        node_to_graph_repr_input = NodesToGraphRepresentationInput(
            node_embeddings=tf.TensorShape((None, final_node_representation_dim)),
            node_to_graph_map=input_shapes["node_to_partial_graph_map"],
            num_graphs=input_shapes["num_graphs_in_batch"],
        )
        with tf.name_scope("latent_representation_computation"):
            with tf.name_scope("weighted_avg"):
                self._weighted_avg_of_nodes_to_graph_repr.build(node_to_graph_repr_input)
            with tf.name_scope("weighted_sum"):
                self._weighted_sum_of_nodes_to_graph_repr.build(node_to_graph_repr_input)

        with tf.name_scope("mean_and_var"):
            self._mean_and_var_mlp.build(latent_graph_representation_shape)

        for prop_name, prop_predictor in self._property_predictors.items():
            with tf.name_scope(f"property_{prop_name}"):
                prop_predictor.build(latent_graph_representation_shape)

        with tf.name_scope("decoder"):
            partial_adjacency_lists: Tuple[tf.TensorShape, ...] = tuple(
                input_shapes[f"partial_adjacency_list_{edge_type_idx}"]
                for edge_type_idx in range(self._num_edge_types)
            )
            self._decoder_layer.build(
                MoLeRDecoderInput(
                    node_features=tf.TensorShape((None, input_shapes["partial_node_features"][-1])),
                    node_categorical_features=tf.TensorShape((None,)),
                    adjacency_lists=partial_adjacency_lists,
                    num_graphs_in_batch=input_shapes["num_partial_graphs_in_batch"],
                    graph_to_focus_node_map=input_shapes["focus_nodes"],
                    node_to_graph_map=input_shapes["node_to_partial_graph_map"],
                    input_molecule_representations=latent_graph_representation_shape,
                    graphs_requiring_node_choices=input_shapes[
                        "partial_graphs_requiring_node_choices"
                    ],
                    candidate_edges=input_shapes["valid_edge_choices"],
                    candidate_edge_features=input_shapes["edge_features"],
                    candidate_attachment_points=input_shapes["valid_attachment_point_choices"],
                )
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

        super().build(input_shapes)

    def get_initial_node_feature_shape(self, input_shapes) -> tf.TensorShape:
        node_features_shape = super().get_initial_node_feature_shape(input_shapes)

        if self.uses_categorical_features:
            node_features_shape = tf.TensorShape(
                dims=(
                    None,
                    node_features_shape[-1] + self._params["categorical_features_embedding_dim"],
                )
            )

        return node_features_shape

    def compute_initial_node_features(self, inputs, training: bool) -> tf.Tensor:
        node_features = super().compute_initial_node_features(inputs, training)

        if self.uses_categorical_features:
            embedded_categorical_features = tf.nn.embedding_lookup(
                self._node_categorical_features_embedding, inputs["node_categorical_features"]
            )

            node_features = tf.concat([node_features, embedded_categorical_features], axis=-1)

        return node_features

    def compute_latent_molecule_representations(
        self,
        final_node_representations: Union[tf.Tensor, Tuple[tf.Tensor, List[tf.Tensor]]],
        num_graphs: tf.Tensor,
        node_to_graph_map: tf.Tensor,
        partial_graph_to_original_graph_map: tf.Tensor,
        training: bool,
    ):
        """Given encoded input graphs, compute representations of them in a latent space
        for a potentially many downstream uses.
        This requires first aggregating the per-node representations into per-graph form,
        and then, dependent on configuration, sampling around the computed per-graph
        representation.

        Sampling strategies:
            passthrough: no noise is added here, and the result will correspond
                deterministically to the input graph representations.
            per_graph: noise is added for each of the input graphs, and then results
                are gathered. Hence, all outputs corresponding to the same graph will
                use the same noise.
            per_partial_graph: results are gathered directly from the input representation,
                and then noise is added to each result independently. This means that
                the all outputs corresponding to the same graph will have different values,
                due to different noise.

        Args:
            final_node_representations:
                The final representations of the graph nodes as computed by the GNN.
                If the hyperparameter "use_intermediate_gnn_results" was set to True,
                a pair of the final node representation and all intermediate node
                representations, including the initial one.
                Shape [V, ED], or pair [V, ED] and list of many [V, ED] tensors; dtype float32.
            num_graphs:
                Number of encoded graphs
            node_to_graph_map:
                A mapping to a graph id for each entry of final_node_representations.
                Shape [V], dtype int32. Values in range [0, ..., num_graphs-1]
            partial_graph_to_original_graph_map:
                A mapping from each partial graph to the ID of the original graph
                corresponding.
                Shape [PG], dtype int32. Values in range [0, ..., num_graphs-1]

        Returns:
            A triple of
                the predicted mean of the graph representations (shape [G, MD]),
                the predicted log variance of the graph representations (shape [G, MD]),
                the representations of the output graphs (shape [PG, MD]).
        """
        if self._params["use_intermediate_gnn_results"]:
            # In this case, we have one initial representation + results for all layers,
            # which we simply collapse into a flat vector:
            final_node_representations = tf.concat(final_node_representations[1], axis=-1)

        node_to_graph_repr_input = NodesToGraphRepresentationInput(
            node_embeddings=final_node_representations,
            node_to_graph_map=node_to_graph_map,
            num_graphs=num_graphs,
        )
        weighted_avg_graph_repr = self._weighted_avg_of_nodes_to_graph_repr(
            node_to_graph_repr_input, training=training
        )
        weighted_sum_graph_repr = self._weighted_sum_of_nodes_to_graph_repr(
            node_to_graph_repr_input, training=training
        )
        input_graph_encodings = tf.concat(
            [weighted_avg_graph_repr, weighted_sum_graph_repr], axis=-1
        )

        graph_mean_and_log_variance = self._mean_and_var_mlp(
            input_graph_encodings, training=training
        )
        graph_mean = graph_mean_and_log_variance[:, : self.latent_dim]  # Shape: [V, MD]
        graph_log_variance = graph_mean_and_log_variance[:, self.latent_dim :]  # Shape: [V, MD]

        if self._latent_sample_strategy == "passthrough":
            result_representations = tf.gather(graph_mean, partial_graph_to_original_graph_map)
        elif self._latent_sample_strategy == "per_graph":
            standard_noise = tf.random.truncated_normal(shape=tf.shape(graph_mean))
            noise = tf.sqrt(tf.exp(graph_log_variance)) * standard_noise
            samples = graph_mean + noise
            result_representations = tf.gather(samples, partial_graph_to_original_graph_map)
        elif self._latent_sample_strategy == "per_partial_graph":
            standard_deviation = tf.sqrt(tf.exp(graph_log_variance))
            partial_graph_mean = tf.gather(graph_mean, partial_graph_to_original_graph_map)
            node_sd = tf.gather(standard_deviation, partial_graph_to_original_graph_map)
            standard_noise = tf.random.truncated_normal(shape=tf.shape(partial_graph_mean))
            noise = node_sd * standard_noise
            result_representations = partial_graph_mean + noise
        else:
            raise ValueError(
                f"Expected sample strategy to be one of "
                f"passthrough, per_graph, or per_partial_graph."
                f"Received: {self._latent_sample_strategy}"
            )

        return graph_mean, graph_log_variance, result_representations

    def compute_task_output(
        self,
        batch_features: Dict[str, tf.Tensor],
        final_node_representations: Union[tf.Tensor, Tuple[tf.Tensor, List[tf.Tensor]]],
        training: bool,
    ) -> MoLeRVaeOutput:
        (
            graph_representation_mean,
            graph_representation_log_variance,
            molecule_representations,
        ) = self.compute_latent_molecule_representations(
            final_node_representations=final_node_representations,
            num_graphs=batch_features["num_graphs_in_batch"],
            node_to_graph_map=batch_features["node_to_graph_map"],
            partial_graph_to_original_graph_map=batch_features[
                "partial_graph_to_original_graph_map"
            ],
            training=training,
        )

        partial_adjacency_lists: Tuple[tf.Tensor, ...] = tuple(
            batch_features[f"partial_adjacency_list_{edge_type_idx}"]
            for edge_type_idx in range(self._num_edge_types)
        )  # Each element has shape (E, 2)
        (
            node_type_logits,
            edge_candidate_logits,
            edge_type_logits,
            attachment_point_selection_logits,
        ) = self._decoder_layer(
            MoLeRDecoderInput(
                node_features=batch_features["partial_node_features"],
                node_categorical_features=batch_features["partial_node_categorical_features"],
                adjacency_lists=partial_adjacency_lists,
                num_graphs_in_batch=batch_features["num_partial_graphs_in_batch"],
                node_to_graph_map=batch_features["node_to_partial_graph_map"],
                graph_to_focus_node_map=batch_features["focus_nodes"],
                input_molecule_representations=molecule_representations,
                graphs_requiring_node_choices=batch_features[
                    "partial_graphs_requiring_node_choices"
                ],
                candidate_edges=batch_features["valid_edge_choices"],
                candidate_edge_features=batch_features["edge_features"],
                candidate_attachment_points=batch_features["valid_attachment_point_choices"],
            ),
            training=training,
        )  # Shape: [PV, NT], [CE + PG, 1], [CE, ET]

        # Property prediction and first node type prediction happen once per input graph (as opposed
        # to once per partial graph). For convenience, get one fresh latent sample per input graph:
        (_, _, per_graph_molecule_representations) = self.compute_latent_molecule_representations(
            final_node_representations=final_node_representations,
            num_graphs=batch_features["num_graphs_in_batch"],
            node_to_graph_map=batch_features["node_to_graph_map"],
            partial_graph_to_original_graph_map=tf.range(0, batch_features["num_graphs_in_batch"]),
            training=training,
        )

        first_node_type_logits = self._decoder_layer.pick_first_node_type(
            input_molecule_representations=per_graph_molecule_representations,
            training=training,
        )

        property_prediction_results: Dict[str, tf.Tensor] = {}
        for prop_name, property_predictor in self._property_predictors.items():
            molecule_with_property_representations = tf.gather(
                per_graph_molecule_representations,
                batch_features[f"graph_property_{prop_name}_graph_ids"],
            )
            property_prediction_results[prop_name] = property_predictor(
                molecule_with_property_representations, training=training
            )

        return MoLeRVaeOutput(
            graph_representation_mean=graph_representation_mean,
            graph_representation_log_variance=graph_representation_log_variance,
            node_type_logits=node_type_logits,
            first_node_type_logits=first_node_type_logits,
            edge_candidate_logits=edge_candidate_logits,
            edge_type_logits=edge_type_logits,
            attachment_point_selection_logits=attachment_point_selection_logits,
            predicted_properties=property_prediction_results,
        )

    def compute_task_metrics(
        self,
        batch_features: Dict[str, tf.Tensor],
        task_output: MoLeRVaeOutput,
        batch_labels: Dict[str, tf.Tensor],
    ) -> Dict[str, tf.Tensor]:
        vae_metrics = self.compute_vae_metrics(batch_features, task_output, batch_labels)

        property_metrics = self.compute_property_predictor_metrics(
            prop_to_predictions=task_output.predicted_properties,
            batch_labels=batch_labels,
        )

        return {
            "loss": vae_metrics.loss + property_metrics.loss,
            "kl_divergence": vae_metrics.kl_divergence,
            "node_classification_loss": vae_metrics.node_classification_loss,
            "first_node_classification_loss": vae_metrics.first_node_classification_loss,
            "edge_loss": vae_metrics.edge_loss,
            "edge_type_loss": vae_metrics.edge_type_loss,
            "attachment_point_selection_loss": vae_metrics.attachment_point_selection_loss,
            "property_to_metrics": property_metrics.property_to_metrics,
        }

    def compute_vae_metrics(
        self,
        batch_features: Dict[str, tf.Tensor],
        task_output: MoLeRVaeOutput,
        batch_labels: Dict[str, tf.Tensor],
    ) -> MoLeRMetrics:

        decoder_metrics = self.decoder.compute_metrics(
            batch_features=batch_features, batch_labels=batch_labels, task_output=task_output
        )

        kl_divergence_summand = (
            tf.square(task_output.graph_representation_mean)
            + tf.exp(task_output.graph_representation_log_variance)
            - task_output.graph_representation_log_variance
            - 1
        )  # Shape: [G, MD]
        kl_divergence_loss = tf.reduce_sum(kl_divergence_summand) / 2.0  # Shape: []

        # Compute KL weight by annealing:
        kl_div_weight = (
            1.0
            - tf.pow(
                self._params["kl_divergence_annealing_beta"],
                tf.cast(self._train_step_counter, dtype=tf.float32),
            )
        ) * self._params["kl_divergence_weight"]
        normalised_kl_loss = kl_divergence_loss / tf.cast(
            batch_features["num_graphs_in_batch"], tf.float32
        )

        total_loss = (
            kl_div_weight * normalised_kl_loss
            + self._params["node_classification_loss_weight"]
            * decoder_metrics.node_classification_loss
            + self._params["first_node_classification_loss_weight"]
            * decoder_metrics.first_node_classification_loss
            + self._params["edge_selection_loss_weight"] * decoder_metrics.edge_loss
            + self._params["edge_type_loss_weight"] * decoder_metrics.edge_type_loss
        )

        if self.uses_motifs:
            total_loss += (
                self._params["attachment_point_selection_weight"]
                * decoder_metrics.attachment_point_selection_loss
            )

        # Don't use 'asdict': that tries to deepcopy decoder_metrics and fails with TypeError
        return MoLeRMetrics(
            loss=total_loss,
            kl_divergence=normalised_kl_loss,
            first_node_classification_loss=decoder_metrics.first_node_classification_loss,
            edge_loss=decoder_metrics.edge_loss,
            edge_type_loss=decoder_metrics.edge_type_loss,
            attachment_point_selection_loss=decoder_metrics.attachment_point_selection_loss,
            node_classification_loss=decoder_metrics.node_classification_loss,
        )

    def compute_property_predictor_metrics(
        self, prop_to_predictions: Dict[str, tf.Tensor], batch_labels: Dict[str, tf.Tensor]
    ) -> PropertyPredictionMetrics:
        total_loss = 0.0
        property_to_metrics: Dict[str, Any] = {}

        for prop_name, prop_predictions in prop_to_predictions.items():
            prop_labels = batch_labels[f"graph_property_{prop_name}_values"]
            loss, pred_metrics = self._property_predictors[prop_name].compute_task_metrics(
                predictions=prop_predictions, labels=prop_labels
            )
            total_loss += self._property_predictor_params[prop_name]["loss_weight_factor"] * loss
            pred_metrics["num_samples"] = tf.shape(prop_labels)[0]
            property_to_metrics[prop_name] = pred_metrics

        return PropertyPredictionMetrics(total_loss, property_to_metrics)

    def compute_epoch_metrics(self, task_results: List[Any]) -> Tuple[float, str]:
        def dict_average(key):
            return np.average([r[key] for r in task_results])

        average_loss = dict_average("loss")
        average_node_classification_loss = dict_average("node_classification_loss")
        average_first_node_classification_loss = dict_average("first_node_classification_loss")
        average_edge_loss = dict_average("edge_loss")
        average_edge_type_loss = dict_average("edge_type_loss")

        if self.uses_motifs:
            average_attachment_point_selection_loss = dict_average(
                "attachment_point_selection_loss"
            )
        else:
            average_attachment_point_selection_loss = None

        average_kl_divergence = dict_average("kl_divergence")

        # Compute results for the individual property predictors.
        # Weigh their respective contributions using the loss weight.
        prop_to_task_results: Dict[str, List[Dict[str, Any]]] = {
            prop_name: [] for prop_name in self._property_predictors.keys()
        }
        for task_result in task_results:
            for prop_name, prop_result in task_result["property_to_metrics"].items():
                prop_to_task_results[prop_name].append(prop_result)
        property_epoch_metric = 0.0
        property_epoch_descriptions = []
        for prop_name, prop_results in prop_to_task_results.items():
            prop_num_samples = np.sum(
                batch_prop_result["num_samples"] for batch_prop_result in prop_results
            )
            prop_metric, prop_metric_description = self._property_predictors[
                prop_name
            ].compute_epoch_metrics(num_samples=prop_num_samples, task_results=prop_results)
            property_epoch_metric += (
                self._property_predictor_params[prop_name]["loss_weight_factor"] * prop_metric
            )
            property_epoch_descriptions.append(f"{prop_name}: {prop_metric_description}")

        graph_generation_losses = [
            ("Avg node class. loss:", f"{average_node_classification_loss: 7.4f}\n"),
            ("Avg first node class. loss:", f"{average_first_node_classification_loss: 7.4f}\n"),
            ("Avg edge selection loss:", f"{average_edge_loss: 7.4f}\n"),
            ("Avg edge type loss:", f"{average_edge_type_loss: 7.4f}\n"),
        ]

        if self.uses_motifs:
            graph_generation_losses.append(
                (
                    "Avg attachment point selection loss:",
                    f"{average_attachment_point_selection_loss: 7.4f}\n",
                )
            )

        graph_generation_losses.append(("Avg KL divergence:", f"{average_kl_divergence: 7.4f}\n"))

        result_string = (
            f"\n"
            f"Avg weighted sum. of graph losses: {average_loss: 7.4f}\n"
            f"Avg weighted sum. of prop losses:  {property_epoch_metric: 7.4f}\n"
        )

        name_column_width = max(len(prefix) for (prefix, _) in graph_generation_losses) + 1

        for prefix, loss in graph_generation_losses:
            # Use extra spaces to allign all graph generation losses.
            result_string += prefix.ljust(name_column_width) + loss

        result_string += f"Property results: {' | '.join(property_epoch_descriptions)}"

        return average_loss + property_epoch_metric, result_string

    def run_on_data_iterator(
        self,
        data_iterator: Iterable[Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]],
        quiet: bool = False,
        training: bool = True,
        max_num_steps: Optional[int] = None,  # Run until dataset ends if None
        aml_run: Optional = None,
    ) -> Tuple[float, float, List[Any]]:
        with EpochMetricsLogger(
            window_size=self._logged_loss_smoothing_window_size,
            quiet=quiet,
            aml_run=aml_run,
            training=training,
        ) as metrics_logger:
            for step, (batch_features, batch_labels) in enumerate(data_iterator):
                if max_num_steps and step >= max_num_steps:
                    break

                task_metrics = self._run_step(batch_features, batch_labels, training)
                metrics_logger.log_step_metrics(task_metrics, batch_features)

        return metrics_logger.get_epoch_summary()
