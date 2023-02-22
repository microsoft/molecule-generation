"""GNN Variational Autoencoder model."""
from typing import Any, Dict, Optional, Tuple, List, NamedTuple, Union, Iterable, DefaultDict, Deque

import numpy as np
import tensorflow as tf
from dpu_utils.tf2utils import MLP
from tf2_gnn import GraphTaskModel
from tf2_gnn.utils import get_activation_function

from molecule_generation.dataset.trace_dataset import TraceDataset
from molecule_generation.layers.cgvae_decoder import CGVAEDecoder, CGVAEDecoderInput
from molecule_generation.layers.graph_property_predictor import (
    GraphPropertyPredictor,
    GraphPropertyPredictorInput,
)
from molecule_generation.utils.training_utils import get_class_balancing_weights
from molecule_generation.chem.atom_feature_utils import AtomFeatureExtractor
from molecule_generation.utils.epoch_metrics_logger import EpochMetricsLogger


class CGVAEOutput(NamedTuple):
    node_representation_mean: tf.Tensor
    node_representation_log_variance: tf.Tensor
    edge_logits: tf.Tensor
    edge_type_logits: tf.Tensor
    node_classification_logits: tf.Tensor
    predicted_properties: Dict[str, tf.Tensor]


class DecoderMetrics(NamedTuple):
    loss: tf.Tensor
    kl_divergence: tf.Tensor
    node_classification_loss: tf.Tensor
    edge_loss: tf.Tensor
    edge_type_loss: tf.Tensor


class PropertyPredictionMetrics(NamedTuple):
    loss: tf.Tensor
    property_to_metrics: Dict[str, Any]


class CGVAE(GraphTaskModel):
    __decoder_prefix = "decoder_"
    __property_predictor_prefix = "property_predictor_"

    @classmethod
    def decoder_prefix(cls) -> str:
        return cls.__decoder_prefix

    @classmethod
    def property_predictor_prefix(cls) -> str:
        return cls.__property_predictor_prefix

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
                "use_intermediate_gnn_results": True,  # Use all intermediate results of the GNN
                "node_classifier_hidden_layers": [64, 32],
                "node_classifier_activation_fun": "relu",
                "node_classifier_dropout_rate": 0.0,
                "sample_strategy": "per_graph",  # one of passthrough, per_graph, or per_partial_graph
                # Property prediction hyperparameters:
                "property_predictor_graph_aggregation_dropout_rate": 0.0,
                "property_predictor_predictor_mlp_dropout_rate": 0.0,
                # Relative contributions of the different losses:
                "node_classification_loss_weight": 1.0,
                "node_classifier_class_loss_weight_factor": 0.1,  # Interpolates between full class weighting (1.0) and no weighting (0.0)
                "edge_selection_loss_weight": 1.0,
                "edge_type_loss_weight": 1.0,
                "kl_divergence_annealing_beta": 0.999,  # 10% weight after 100 steps, 63% after 1000, 99% after 5000
                "kl_divergence_weight": 0.3,
                # Training hyperparameters:
                "num_train_steps_between_valid": 0,  # Default value of 0 means that whole dataset will be run through.
                # Not really hyperparameters, but logging parameters:
                "logged_loss_smoothing_window_size": 100,  # Number of training batches to include in the reported moving average training loss.
            }
        )

        # Add hyperparameters for the GNN decoder:
        decoder_hypers = {
            cls.decoder_prefix() + k: v
            for k, v in CGVAEDecoder.get_default_params(mp_style).items()
        }
        base_hypers.update(decoder_hypers)

        # Add hyperparameters for the (co-trained) property predictors:
        property_predictor_hypers = {
            cls.property_predictor_prefix() + k: v
            for k, v in GraphPropertyPredictor.get_default_hyperparameters().items()
        }
        base_hypers.update(property_predictor_hypers)

        return base_hypers

    def __init__(self, params: Dict[str, Any], dataset: TraceDataset, **kwargs):
        super().__init__(params, dataset, **kwargs)

        # Shortcuts to commonly used hyperparameters:
        self._latent_repr_dim = params["gnn_hidden_dim"]
        self._sample_strategy = params["sample_strategy"]

        # Get some information out from the dataset:
        self._node_type_index_to_string: Dict[int, str] = dataset.node_type_index_to_string
        self._graph_property_params = dataset.params["graph_properties"]
        self._feature_extractors: List[AtomFeatureExtractor] = dataset.metadata.get(
            "feature_extractors", []
        )
        self._num_node_types = dataset.num_node_types
        self._decoder_use_self_loop_edges_in_partial_graphs = dataset.params["add_self_loop_edges"]
        self._graph_property_metadata = {}
        for prop_name in self._graph_property_params.keys():
            self._graph_property_metadata[f"{prop_name}_stddev"] = dataset.metadata.get(
                f"{prop_name}_stddev"
            )
        self._atom_type_distribution = dataset.metadata.get("train_atom_type_distribution")

        # ===== Prepare sub-layers, which will be actually created in .build()
        self._mean_and_var_mlp = MLP(
            out_size=self._latent_repr_dim * 2,
            hidden_layers=0,
            use_biases=True,
            name="MeanAndVarMLP",
        )

        self._node_to_label_layer = MLP(
            out_size=self._num_node_types,
            hidden_layers=self._params["node_classifier_hidden_layers"],
            use_biases=True,
            activation_fun=get_activation_function(self._params["node_classifier_activation_fun"]),
            dropout_rate=self._params["node_classifier_dropout_rate"],
        )

        decoder_prefix = self.decoder_prefix()
        n = len(decoder_prefix)
        decoder_hypers = {k[n:]: v for k, v in self._params.items() if k.startswith(decoder_prefix)}
        class_weight_factor = self._params.get("node_classifier_class_loss_weight_factor", 0.0)
        if not (0.0 <= class_weight_factor <= 1.0):
            raise ValueError(
                f"Node class loss weight node_classifier_class_loss_weight_factor must be in [0,1], but is {class_weight_factor}!"
            )
        if class_weight_factor > 0 and self._atom_type_distribution is not None:
            atom_type_nums = [
                self._atom_type_distribution[self._node_type_index_to_string[type_idx]]
                for type_idx in range(self._num_node_types)
            ]

            class_weights = get_class_balancing_weights(
                class_counts=atom_type_nums, class_weight_factor=class_weight_factor
            )
        else:
            class_weights = None
        self._decoder_layer = CGVAEDecoder(
            decoder_hypers,
            name="CGVAEDecoder",
            use_self_loop_edges_in_partial_graphs=self._decoder_use_self_loop_edges_in_partial_graphs,
            feature_extractors=self._feature_extractors,
            node_type_loss_weights=class_weights,
        )

        self._property_predictors: Dict[str, GraphPropertyPredictor] = {}
        n = len(self.property_predictor_prefix())
        property_predictor_default_hypers = {
            k[n:]: v
            for k, v in self._params.items()
            if k.startswith(self.property_predictor_prefix())
        }
        for prop_name, prop_params in self._graph_property_params.items():
            property_predictor_hypers = dict(property_predictor_default_hypers)
            property_predictor_hypers.update(prop_params)
            with tf.name_scope(f"{prop_name}_prediction"):
                property_stddev = self._graph_property_metadata[f"{prop_name}_stddev"]
                if not (prop_params.get("normalise_loss", True)):
                    property_stddev = None
                self._property_predictors[prop_name] = GraphPropertyPredictor(
                    params=property_predictor_hypers,
                    property_type=prop_params["type"],
                    property_stddev=property_stddev,
                )

        # Moving average variables, will be filled later:
        self._logged_loss_smoothing_window_size = self._params["logged_loss_smoothing_window_size"]
        self._smoothed_stats_raw_values: DefaultDict[str, Deque[Union[float, int]]] = None

        # Deal with Tensorboard's global state:
        tf.summary.experimental.set_step(0)

    @property
    def latent_dim(self):
        return self._latent_repr_dim

    @property
    def decoder(self):
        return self._decoder_layer

    def build(self, input_shapes: Dict[str, Any]):
        encoder_gnn_dim = self._params["gnn_hidden_dim"]
        if self._params["use_intermediate_gnn_results"]:
            # We get the initial GNN input (after projection) + results for all layers:
            final_node_representation_dim = encoder_gnn_dim * (1 + self._params["gnn_num_layers"])
            final_node_representation_spec: Tuple[tf.TensorSpec, ...] = (
                tf.TensorSpec(
                    shape=(None, encoder_gnn_dim), dtype=tf.float32
                ),  # the actual final reprs
                tuple(
                    tf.TensorSpec(
                        shape=(None, encoder_gnn_dim), dtype=tf.float32
                    )  # the per per-layer reprs
                    for _ in range(1 + self._params["gnn_num_layers"])
                ),
            )
        else:
            final_node_representation_dim = encoder_gnn_dim
            final_node_representation_spec = (
                tf.TensorSpec(shape=(None, encoder_gnn_dim), dtype=tf.float32),
            )  # the actual final reprs
        encoded_node_representation_shape = tf.TensorShape(
            dims=(None, final_node_representation_dim)
        )
        # We need to use setattr here instead of just using the decorator to account
        # for the Union type of the first argument...
        setattr(
            self,
            "get_node_samples",
            tf.function(
                input_signature=(
                    final_node_representation_spec,
                    tf.TensorSpec(shape=tf.TensorShape((None,)), dtype=tf.int32),
                    tf.TensorSpec(shape=tf.TensorShape((None, None)), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.bool),
                )
            )(self.get_node_samples),
        )

        with tf.name_scope("mean_and_var"):
            self._mean_and_var_mlp.build(encoded_node_representation_shape)

        latent_node_representation_shape = tf.TensorShape(dims=(None, self._latent_repr_dim))
        partial_node_representation_shape = tf.TensorShape(
            dims=(None, self._latent_repr_dim + input_shapes["partial_node_features"][-1])
        )
        with tf.name_scope("node_classifier"):
            self._node_to_label_layer.build(latent_node_representation_shape)

        with tf.name_scope("decoder"):
            partial_adjacency_lists: Tuple[tf.TensorShape, ...] = tuple(
                input_shapes[f"partial_adjacency_list_{edge_type_idx}"]
                for edge_type_idx in range(self._num_edge_types)
            )
            decoder_input = CGVAEDecoderInput(
                node_features=partial_node_representation_shape,
                adjacency_lists=partial_adjacency_lists,
                num_partial_graphs_in_batch=input_shapes["num_partial_graphs_in_batch"],
                graph_to_focus_node_map=input_shapes["focus_nodes"],
                node_to_graph_map=input_shapes["node_to_partial_graph_map"],
                valid_edge_choices=input_shapes["valid_edge_choices"],
                edge_features=input_shapes["edge_features"],
            )
            self._decoder_layer.build(decoder_input)

        prop_predictor_input = GraphPropertyPredictorInput(
            node_representations=latent_node_representation_shape,
            node_to_graph_map=input_shapes["node_to_partial_graph_map"],
            num_graphs=input_shapes["num_graphs_in_batch"],
            graph_ids_to_predict_for=tf.TensorShape(dims=(None,)),
        )
        for prop_name, prop_predictor in self._property_predictors.items():
            with tf.name_scope(f"{prop_name}_prediction"):
                prop_predictor.build(prop_predictor_input)

        super().build(input_shapes)

    def compute_node_mean_and_logvariance(
        self,
        node_representations: Union[tf.Tensor, Tuple[tf.Tensor, List[tf.Tensor]]],
        training: bool,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        if self._params["use_intermediate_gnn_results"]:
            # We get the initial GNN input (after projection) + results for all layers:
            node_representations = tf.concat(
                node_representations[1], axis=-1
            )  # Shape [V, ED*(num_layers+1)]

        node_mean_and_log_var = self._mean_and_var_mlp(node_representations, training=training)
        node_mean = node_mean_and_log_var[:, : self._latent_repr_dim]  # Shape: [V, ED]
        node_log_variance = node_mean_and_log_var[:, self._latent_repr_dim :]  # Shape: [V, ED]

        return node_mean, node_log_variance

    @tf.function(experimental_relax_shapes=True)
    def generate_samples(
        self,
        node_mean: tf.Tensor,
        node_log_variance: tf.Tensor,
        partial_node_to_original_node_map: tf.Tensor,
    ) -> tf.Tensor:
        """Generate random samples around the node mean, using one of three sampling strategies.

        Args:
            node_mean: The (full graph) node mean. Shape: [V, ED]
            node_log_variance: The (full graph) node logvariance. Shape: [V, ED]
            partial_node_to_original_node_map: The map from partial node idx to original node idx.

        Returns:
            Sampled node embeddings for each node in the partial graphs. A tensor of shape [PV, ED].

        Sampling strategies:
            passthrough: no random sampling is performed here. The node mean from the full graph
                are gathered to the partial nodes.
            per_graph: one sample is generated for each of the nodes in the original graphs, and
                then gathered to the partial nodes.
            per_partial_graph: the node means and variances are gathered to each of the partial
                graphs, and then one sample is generated for each one of those.

        """
        if self._sample_strategy == "passthrough":
            return tf.gather(node_mean, partial_node_to_original_node_map)
        elif self._sample_strategy == "per_graph":
            return self._generate_per_graph_samples(
                node_mean, node_log_variance, partial_node_to_original_node_map
            )
        elif self._sample_strategy == "per_partial_graph":
            return self._generate_per_partial_graph_samples(
                node_mean, node_log_variance, partial_node_to_original_node_map
            )
        else:
            raise ValueError(
                f"Expected sample strategy to be one of "
                f"passthrough, per_graph, or per_partial_graph."
                f"Received: {self._sample_strategy}"
            )

    def _generate_per_graph_samples(
        self, mean: tf.Tensor, logvariance: tf.Tensor, node_map: tf.Tensor
    ) -> tf.Tensor:
        standard_noise = tf.random.normal(shape=tf.shape(mean))
        noise = tf.sqrt(tf.exp(logvariance)) * standard_noise
        samples = mean + noise
        return tf.gather(samples, node_map)

    def _generate_per_partial_graph_samples(
        self, mean: tf.Tensor, logvariance: tf.Tensor, node_map: tf.Tensor
    ) -> tf.Tensor:
        standard_deviation = tf.sqrt(tf.exp(logvariance))
        node_mean = tf.gather(mean, node_map)
        node_sd = tf.gather(standard_deviation, node_map)
        standard_noise = tf.random.normal(shape=tf.shape(node_mean))
        noise = node_sd * standard_noise
        return node_mean + noise

    def classify_nodes(self, node_features: tf.Tensor) -> List[str]:
        """Classify the nodes based on their features. One hot encoded output.

        Args:
            node_features: a tensor of node features, with shape [V, ED], where V is the number
                of nodes to be classified, and ED is the embedding dimension.

        Returns:
            A list of string representations of the node types.

        """
        node_classification_logits = self._node_to_label_layer(
            node_features, training=False
        )  # shape: [V, NT]
        # We never want atom type "UNK" (index 0), so remove that value. However, the indexing still
        # accounts for it, so add it back in the lookup:
        max_indices = tf.argmax(node_classification_logits[:, 1:], axis=1)  # shape: [V]
        return [self._node_type_index_to_string[1 + index.numpy()] for index in max_indices]

    def get_node_samples(
        self,
        final_node_representations: Union[tf.Tensor, Tuple[tf.Tensor, List[tf.Tensor]]],
        partial_node_to_original_node_map: tf.Tensor,
        partial_graph_node_features: tf.Tensor,
        training: bool,
    ):
        node_mean, node_log_variance = self.compute_node_mean_and_logvariance(
            final_node_representations, training
        )

        # We need to generate two different kinds of node samples here. First, we generate
        # one sample per node in the target graph, to be used for node classification and
        # graph property prediction. Second, we generate one sample per node in all the
        # partial graphs, which contain (potentially many) copies of nodes in the final graphs.
        node_samples = self.generate_samples(
            node_mean, node_log_variance, tf.range(tf.shape(node_mean)[0])
        )

        partial_graph_node_samples = self.generate_samples(
            node_mean, node_log_variance, partial_node_to_original_node_map
        )  # Shape: [PV, ED]

        partial_graph_node_representations = tf.concat(
            [partial_graph_node_samples, partial_graph_node_features], axis=-1
        )

        return node_mean, node_log_variance, node_samples, partial_graph_node_representations

    def compute_task_output(
        self,
        batch_features: Dict[str, tf.Tensor],
        final_node_representations: Union[tf.Tensor, Tuple[tf.Tensor, List[tf.Tensor]]],
        training: bool,
    ) -> CGVAEOutput:
        """Computes the task output for the gnn vae.

        We use the following abbreviations for the shapes:
            - V: number of vertices in this batch
            - ED: encoded / hidden dimension of each vertex
            - PV: number of vertices in the partial graphs in this batch
            - PG: number of partial graphs in this batch
            - VE: number of valid edges in this batch
            - E: number of edges of a given type (can be different per type)
            - ET: number of edge types for this task
            - NT: number of node types for this task
        """
        (
            node_mean,
            node_log_variance,
            node_samples,
            partial_graph_node_representations,
        ) = self.get_node_samples(
            final_node_representations,
            batch_features["partial_node_to_original_node_map"],
            batch_features["partial_node_features"],
            training,
        )

        node_classification_logits = self._node_to_label_layer(node_samples)

        partial_adjacency_lists: Tuple[tf.Tensor, ...] = tuple(
            batch_features[f"partial_adjacency_list_{edge_type_idx}"]
            for edge_type_idx in range(self._num_edge_types)
        )  # Each element has shape (E, 2)
        decoder_input = CGVAEDecoderInput(
            node_features=partial_graph_node_representations,
            adjacency_lists=partial_adjacency_lists,
            num_partial_graphs_in_batch=batch_features["num_partial_graphs_in_batch"],
            graph_to_focus_node_map=batch_features["focus_nodes"],
            node_to_graph_map=batch_features["node_to_partial_graph_map"],
            valid_edge_choices=batch_features["valid_edge_choices"],
            edge_features=batch_features["edge_features"],
        )

        edge_logits, edge_type_logits = self._decoder_layer(
            decoder_input, training=training
        )  # Shape: [VE + PG, 1], [PV, ET], [PV, NT]

        property_prediction_results: Dict[str, tf.Tensor] = {}
        for prop_name, property_predictor in self._property_predictors.items():
            prop_prediction_input = GraphPropertyPredictorInput(
                node_representations=node_samples,
                node_to_graph_map=batch_features["node_to_graph_map"],
                num_graphs=batch_features["num_graphs_in_batch"],
                graph_ids_to_predict_for=batch_features[f"graph_property_{prop_name}_graph_ids"],
            )
            property_prediction_results[prop_name] = property_predictor(
                prop_prediction_input, training=training
            )

        return CGVAEOutput(
            node_representation_mean=node_mean,
            node_representation_log_variance=node_log_variance,
            edge_logits=edge_logits,
            edge_type_logits=edge_type_logits,
            node_classification_logits=node_classification_logits,
            predicted_properties=property_prediction_results,
        )

    def compute_task_metrics(
        self,
        batch_features: Dict[str, tf.Tensor],
        task_output: CGVAEOutput,
        batch_labels: Dict[str, tf.Tensor],
    ) -> Dict[str, tf.Tensor]:
        """Compute the loss and related metrics for this task.

        We use the same shape abbreviations as in the compute task output method, along with
        - VE: number of valid edges in this batch.
        - CE: number of correct edge choices for this batch
        """
        vae_metrics = self.compute_vae_metrics(batch_features, task_output, batch_labels)

        property_metrics = self.compute_property_predictor_metrics(
            prop_to_predictions=task_output.predicted_properties,
            batch_labels=batch_labels,
        )

        return {
            "loss": vae_metrics.loss + property_metrics.loss,
            "kl_divergence": vae_metrics.kl_divergence,
            "node_classification_loss": vae_metrics.node_classification_loss,
            "edge_loss": vae_metrics.edge_loss,
            "edge_type_loss": vae_metrics.edge_type_loss,
            "property_to_metrics": property_metrics.property_to_metrics,
        }

    def compute_vae_metrics(
        self,
        batch_features: Dict[str, tf.Tensor],
        task_output: CGVAEOutput,
        batch_labels: Dict[str, tf.Tensor],
    ) -> DecoderMetrics:
        num_graphs_in_batch = tf.cast(batch_features["num_graphs_in_batch"], dtype=tf.float32)
        # KL divergence.
        kl_divergence_summand = (
            tf.square(task_output.node_representation_mean)
            + tf.exp(task_output.node_representation_log_variance)
            - task_output.node_representation_log_variance
            - 1
        )  # Shape: [V, ED]
        kl_divergence = tf.reduce_sum(kl_divergence_summand) / (
            2 * num_graphs_in_batch
        )  # Shape: []

        (
            edge_loss,
            edge_type_loss,
            node_classification_loss,
        ) = self._decoder_layer.calculate_reconstruction_loss(
            node_type_logits=task_output.node_classification_logits,
            edge_logits=task_output.edge_logits,
            edge_type_logits=task_output.edge_type_logits,
            node_type_label_indices=batch_labels["node_types"],  # Shape: [V]
            num_graphs_in_batch=num_graphs_in_batch,
            num_partial_graphs_in_batch=batch_features["num_partial_graphs_in_batch"],
            node_to_partial_graph_map=batch_features["node_to_partial_graph_map"],  # Shape: [PV]
            correct_target_node_multihot=batch_labels["correct_edge_choices"],  # Shape: [VE]
            valid_target_node_idx=batch_features["valid_edge_choices"][:, 1],  # Shape: [VE]
            stop_node_label=batch_labels["stop_node_label"],  # Shape: [PG]
            num_correct_edge_choices=batch_labels["num_correct_edge_choices"],  # Shape: [PG]
            one_hot_edge_types=batch_labels["correct_edge_types"],  # Shape: [CE, ET]
            valid_edge_types=batch_labels["valid_edge_types"],  # Shape: [CE, ET]
        )

        # Compute KL weight by annealing:
        kl_div_weight = (
            1.0
            - tf.pow(
                self._params["kl_divergence_annealing_beta"],
                tf.cast(self._train_step_counter, dtype=tf.float32),
            )
        ) * self._params["kl_divergence_weight"]

        # Total loss.
        total_loss = (
            kl_div_weight * kl_divergence
            + self._params["node_classification_loss_weight"] * node_classification_loss
            + self._params["edge_selection_loss_weight"] * edge_loss
            + self._params["edge_type_loss_weight"] * edge_type_loss
        )
        return DecoderMetrics(
            loss=total_loss,
            kl_divergence=kl_divergence,
            node_classification_loss=node_classification_loss,
            edge_loss=edge_loss,
            edge_type_loss=edge_type_loss,
        )

    def compute_property_predictor_metrics(
        self, prop_to_predictions: Dict[str, tf.Tensor], batch_labels: Dict[str, tf.Tensor]
    ) -> PropertyPredictionMetrics:
        total_loss = 0.0
        property_to_metrics: Dict[str, Any] = {}

        for prop_name, prop_predictions in prop_to_predictions.items():
            prop_params = self._graph_property_params[prop_name]
            prop_labels = batch_labels[f"graph_property_{prop_name}_values"]
            loss, pred_metrics = self._property_predictors[prop_name].compute_task_metrics(
                property_predictions=prop_predictions, property_labels=prop_labels
            )
            total_loss += prop_params["loss_weight_factor"] * loss
            property_to_metrics[prop_name] = pred_metrics

        return PropertyPredictionMetrics(total_loss, property_to_metrics)

    def compute_epoch_metrics(self, task_results: List[Any]) -> Tuple[float, str]:
        def dict_average(key):
            return np.average([r[key] for r in task_results])

        average_loss = dict_average("loss")
        average_node_classification_loss = dict_average("node_classification_loss")
        average_edge_loss = dict_average("edge_loss")
        average_edge_type_loss = dict_average("edge_type_loss")
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
            prop_metric, prop_metric_description = self._property_predictors[
                prop_name
            ].compute_epoch_metrics(prop_results)
            property_epoch_metric += (
                self._graph_property_params[prop_name]["loss_weight_factor"] * prop_metric
            )
            property_epoch_descriptions.append(f"{prop_name}: {prop_metric_description}")

        result_string = (
            f"\n"
            f"Avg node class. loss: {average_node_classification_loss}\n"
            f"Avg edge loss: {average_edge_loss}\n"
            f"Avg edge type loss: {average_edge_type_loss}\n"
            f"Avg KL divergence: {average_kl_divergence}\n"
            f"Property results: {' | '.join(property_epoch_descriptions)}"
        )
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
