"""MoLeR Variational Autoencoder model."""
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, NamedTuple, Union

import numpy as np
import tensorflow as tf
from dpu_utils.tf2utils import MLP
from tf2_gnn import GraphTaskModel
from tf2_gnn.layers import NodesToGraphRepresentationInput, WeightedSumGraphRepresentation

from molecule_generation.dataset.trace_dataset import TraceDataset
from molecule_generation.layers.moler_decoder import MoLeRDecoderMetrics
from molecule_generation.models.moler_base_model import MoLeRBaseModel
from molecule_generation.utils.property_models import (
    PropertyTaskType,
    PropertyPredictionLayer,
    MLPBinaryClassifierLayer,
    MLPRegressionLayer,
)


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


class MoLeRVae(MoLeRBaseModel):
    """[Mo]lecule [Le]vel [R]representation-based VAE model for handling molecular graphs.

    The components of our model work together as follows:
        Molecule -> Encoder -> NodeToGraphRepresentation -> Sampling -> MoLeRDecoder -> Molecule

    Molecules are built both using single atoms and larger chunks (motifs).
    The vocabulary of motifs has to be precomputed during preprocessing.
    """

    @classmethod
    def get_default_hyperparameters(cls, mp_style: Optional[str] = None) -> Dict[str, Any]:
        base_hypers = super().get_default_hyperparameters(mp_style)

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
                "latent_repr_mlp_layers": [512, 512],
                "latent_repr_num_heads": 32,
                "latent_repr_dropout_rate": 0.0,
                "latent_sample_strategy": "per_graph",  # one of passthrough, per_graph, or per_partial_graph
                # Property prediction hyperparameters:
                "property_predictor_mlp_layers": [64, 32],
                "property_predictor_mlp_dropout_rate": 0.0,
                # Relative contribution of the KL loss term. This was tuned for a motif
                # vocabulary of 128, may need to be adjusted for larger vocabularies
                # (empirically increments of 0.005 every time vocabulary size doubles):
                "kl_divergence_weight": 0.02,
                "kl_divergence_annealing_beta": 0.999,  # 10% weight after 100 steps, 63% after 1000, 99% after 5000
            }
        )

        return base_hypers

    def __init__(self, params: Dict[str, Any], dataset: TraceDataset, **kwargs):
        super().__init__(params, dataset, **kwargs)

        self._latent_sample_strategy = params["latent_sample_strategy"]

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

    def build(self, input_shapes: Dict[str, Any]):

        # Build decoder
        super().build(input_shapes=input_shapes)

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

        # Build encoder and mark self as built
        GraphTaskModel.build(self, input_shapes)

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

        # result_representations: shape [PG, MD]
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

        decoder_output = self._get_decoder_output(
            batch_features=batch_features,
            molecule_representations=molecule_representations,
            training=training,
        )

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
            node_type_logits=decoder_output.node_type_logits,
            first_node_type_logits=first_node_type_logits,
            edge_candidate_logits=decoder_output.edge_candidate_logits,
            edge_type_logits=decoder_output.edge_type_logits,
            attachment_point_selection_logits=decoder_output.attachment_point_selection_logits,
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

        total_loss, decoder_metrics = self._compute_decoder_loss_and_metrics(
            batch_features=batch_features, task_output=task_output, batch_labels=batch_labels
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

        total_loss += kl_div_weight * normalised_kl_loss

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

    def _get_graph_generation_losses(self, task_results: List[Any]) -> List[Tuple[str, str]]:
        graph_generation_losses = super()._get_graph_generation_losses(task_results)
        average_kl_divergence = self._dict_average(task_results, "kl_divergence")
        graph_generation_losses.append(("Avg KL divergence:", f"{average_kl_divergence: 7.4f}\n"))
        return graph_generation_losses

    def compute_epoch_metrics(self, task_results: List[Any]) -> Tuple[float, str]:

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

        average_loss = self._dict_average(task_results, "loss")
        result_string = (
            f"\n"
            f"Avg weighted sum. of graph losses: {average_loss: 7.4f}\n"
            f"Avg weighted sum. of prop losses:  {property_epoch_metric: 7.4f}\n"
        )

        graph_generation_losses = self._get_graph_generation_losses(task_results)
        result_string += self._format_graph_generation_losses(graph_generation_losses)

        result_string += f"Property results: {' | '.join(property_epoch_descriptions)}"

        return average_loss + property_epoch_metric, result_string
