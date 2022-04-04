"""Task handling prediction of multiple properties for graphs."""
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional, Union, Callable

import numpy as np
import tensorflow as tf
from tf2_gnn.data import GraphDataset
from tf2_gnn.models import GraphTaskModel
from tf2_gnn.layers import WeightedSumGraphRepresentation, NodesToGraphRepresentationInput

from molecule_generation.dataset.jsonl_graph_properties_dataset import PropertyTaskType
from molecule_generation.utils.property_models import (
    MLPBinaryClassifierLayer,
    MLPRegressionLayer,
)


class GraphMultitaskModel(GraphTaskModel):
    @classmethod
    def get_default_hyperparameters(cls, mp_style: Optional[str] = None) -> Dict[str, Any]:
        super_params = super().get_default_hyperparameters(mp_style)
        these_hypers: Dict[str, Any] = {
            "graph_aggregation_weighting_fun": "sigmoid",
            "graph_representation_size": 64,
            "graph_aggregation_num_heads": 16,
            "graph_aggregation_hidden_layers": [64, 64],
            "graph_aggregation_dropout_rate": 0.2,
            "use_intermediate_gnn_results": True,
            # If true, each task uses a different instantiation of the graph aggregation model
            # to compute a graph representation; otherwise all share the same graph representation
            "separate_graph_aggregation_layers": True,
            "task_mlp_layer_sizes": [32, 32],
            "task_mlp_dropout_rate": 0.2,
        }
        super_params.update(these_hypers)
        return super_params

    def __init__(self, params: Dict[str, Any], dataset: GraphDataset, name: str = None, **kwargs):
        super().__init__(params, dataset=dataset, name=name, **kwargs)
        self._task_names = dataset._property_names
        self._task_types: Dict[str, PropertyTaskType] = dataset._property_name_to_type
        self._graph_aggregation_layers: Dict[str, tf.keras.layers.Layer] = {}
        self._graph_task_layers: Dict[str, tf.keras.layers.Layer] = {}

    def __build_graph_aggregation_layer(self, input_shapes):
        node_to_graph_repr_layer = WeightedSumGraphRepresentation(
            weighting_fun=self._params["graph_aggregation_weighting_fun"],
            graph_representation_size=self._params["graph_representation_size"],
            num_heads=self._params["graph_aggregation_num_heads"],
            scoring_mlp_layers=self._params["graph_aggregation_hidden_layers"],
            scoring_mlp_dropout_rate=self._params["graph_aggregation_dropout_rate"],
            transformation_mlp_layers=self._params["graph_aggregation_hidden_layers"],
            transformation_mlp_dropout_rate=self._params["graph_aggregation_dropout_rate"],
        )

        if self._params["use_intermediate_gnn_results"]:
            # Representation is initial representation + one output per layer:
            node_representation_size = (
                input_shapes["node_features"][-1]
                + self._params["gnn_num_layers"] * self._params["gnn_hidden_dim"]
            )
        else:
            # Representation is initial representation + final output:
            node_representation_size = (
                input_shapes["node_features"][-1] + self._params["gnn_hidden_dim"]
            )

        node_to_graph_repr_layer.build(
            NodesToGraphRepresentationInput(
                node_embeddings=tf.TensorShape((None, node_representation_size)),
                node_to_graph_map=tf.TensorShape((None)),
                num_graphs=tf.TensorShape(()),
            )
        )
        return node_to_graph_repr_layer

    def build(self, input_shapes):
        if not self._params["separate_graph_aggregation_layers"]:
            self._graph_aggregation_layers["SHARED"] = self.__build_graph_aggregation_layer(
                input_shapes
            )

        for task_name in self._task_names:
            task_type = self._task_types[task_name]
            with tf.name_scope(f"Task_{task_name}"):
                if self._params["separate_graph_aggregation_layers"]:
                    self._graph_aggregation_layers[
                        task_name
                    ] = self.__build_graph_aggregation_layer(input_shapes)

                if task_type == PropertyTaskType.BINARY_CLASSIFICATION:
                    task_model = MLPBinaryClassifierLayer(
                        mlp_layer_sizes=self._params["task_mlp_layer_sizes"],
                        dropout_rate=self._params["task_mlp_dropout_rate"],
                    )
                elif task_type == PropertyTaskType.REGRESSION:
                    task_model = MLPRegressionLayer(
                        mlp_layer_sizes=self._params["task_mlp_layer_sizes"],
                        dropout_rate=self._params["task_mlp_dropout_rate"],
                    )
                else:
                    raise ValueError(f"Unknown task type {task_type}")
                task_model.build(tf.TensorShape((None, self._params["graph_representation_size"])))
                self._graph_task_layers[task_name] = task_model

        super().build(input_shapes)

    def compute_task_output(
        self,
        batch_features: Dict[str, tf.Tensor],
        final_node_representations: Union[tf.Tensor, Tuple[tf.Tensor, List[tf.Tensor]]],
        training: bool,
    ) -> Any:
        if self._params["use_intermediate_gnn_results"]:
            _, intermediate_node_representations = final_node_representations
            # We want to skip the first "intermediate" representation, which is the output of
            # the initial feature -> GNN input layer:
            node_representations = tf.concat(
                [batch_features["node_features"]] + intermediate_node_representations[1:], axis=-1
            )
        else:
            node_representations = tf.concat(
                [batch_features["node_features"], final_node_representations], axis=-1
            )

        graph_aggregation_input = NodesToGraphRepresentationInput(
            node_embeddings=node_representations,
            node_to_graph_map=batch_features["node_to_graph_map"],
            num_graphs=batch_features["num_graphs_in_batch"],
        )

        if not self._params["separate_graph_aggregation_layers"]:
            graph_representations = self._graph_aggregation_layers["SHARED"](
                graph_aggregation_input, training=training
            )  # Shape [G, graph_representation_size]

        per_task_results = {}
        for task_name in self._task_names:
            if self._params["separate_graph_aggregation_layers"]:
                graph_representations = self._graph_aggregation_layers[task_name](
                    graph_aggregation_input, training=training
                )  # Shape [G, graph_representation_size]
            task_results = self._graph_task_layers[task_name](
                graph_representations, training=training
            )
            per_task_results[task_name] = task_results

        return per_task_results

    def compute_task_metrics(
        self,
        batch_features: Dict[str, tf.Tensor],
        task_output: Any,
        batch_labels: Dict[str, tf.Tensor],
    ) -> Dict[str, Any]:
        num_graphs = tf.cast(batch_features["num_graphs_in_batch"], tf.float32)
        results = {"loss": 0.0, "num_graphs": num_graphs, "per_task_results": {}}
        for task_idx, task_name in enumerate(self._task_names):
            predictions = task_output[task_name]  # Shape [G]
            labels = batch_labels["target_values"][:, task_idx]  # Shape [G]
            loss, batch_results = self._graph_task_layers[task_name].compute_task_metrics(
                predictions=predictions, labels=labels
            )
            results["loss"] += loss
            results["per_task_results"][task_name] = batch_results
        return results

    def compute_epoch_metrics(self, task_results: List[Any]) -> Tuple[float, str]:
        total_num_graphs = np.sum(
            batch_task_result["num_graphs"] for batch_task_result in task_results
        )

        task_fitness_values = []
        task_strs = []
        for task_idx, task_name in enumerate(self._task_names):
            task_fitness, task_str = self._graph_task_layers[task_name].compute_epoch_metrics(
                num_samples=total_num_graphs,
                task_results=[
                    task_result["per_task_results"][task_name] for task_result in task_results
                ],
            )
            task_fitness_values.append(task_fitness)
            task_strs.append(f"{task_name}: {task_str}")

        return np.mean(task_fitness_values), ", ".join(task_strs)

    def evaluate(self, dataset: tf.data.Dataset, log_fun: Callable[[str], None] = print) -> None:
        task_to_labels: Dict[str, List[np.ndarray]] = defaultdict(list)
        task_to_predictions: Dict[str, List[np.ndarray]] = defaultdict(list)
        for batch_features, batch_labels in dataset:
            outputs = self(batch_features, training=False)
            for task_idx, task_name in enumerate(self._task_names):
                predictions = outputs[task_name]  # Shape [G]
                labels = batch_labels["target_values"][:, task_idx]  # Shape [G]
                task_to_predictions[task_name].append(predictions)
                task_to_labels[task_name].append(labels)

        for task_name, task_type in sorted(self._task_types.items(), key=lambda kv: kv[0]):
            task_labels = np.concatenate(task_to_labels[task_name], axis=0)
            task_predictions = np.concatenate(task_to_predictions[task_name], axis=0)

            if task_type == PropertyTaskType.BINARY_CLASSIFICATION:
                MLPBinaryClassifierLayer.print_evaluation_report(
                    task_name,
                    predictions=task_predictions,
                    labels=task_labels,
                    log_fun=log_fun,
                )
            elif task_type == PropertyTaskType.BINARY_CLASSIFICATION:
                MLPRegressionLayer.print_evaluation_report(
                    task_name,
                    predictions=task_predictions,
                    labels=task_labels,
                    log_fun=log_fun,
                )
