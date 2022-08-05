"""Task handling prediction of a property from graphs represented by individual
node representations."""
from typing import Any, Dict, List, Tuple, NamedTuple, Optional

import numpy as np
import tensorflow as tf
from tf2_gnn.layers import WeightedSumGraphRepresentation, NodesToGraphRepresentationInput

from molecule_generation.utils.property_models import (
    PropertyTaskType,
    MLPBinaryClassifierLayer,
    MLPRegressionLayer,
)


class GraphPropertyPredictorInput(NamedTuple):
    node_representations: tf.Tensor
    node_to_graph_map: tf.Tensor
    num_graphs: tf.Tensor
    graph_ids_to_predict_for: tf.Tensor


class GraphPropertyPredictor(tf.keras.layers.Layer):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        return {
            "graph_representation_size": 64,
            "graph_aggregation_weighting_fun": "softmax",
            "graph_aggregation_num_heads": 16,  # Needs to divide graph_representation_size
            "graph_aggregation_hidden_layers": [64, 64],
            "graph_aggregation_dropout_rate": 0.2,
            "predictor_mlp_layer_sizes": [32, 32],
            "predictor_mlp_dropout_rate": 0.2,
        }

    def __init__(
        self,
        params: Dict[str, Any],
        property_type: str,
        property_stddev: Optional[float],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._params = params
        self._property_type = PropertyTaskType[property_type.upper()]

        with tf.name_scope(self._name):
            self._graph_aggregation_layer = WeightedSumGraphRepresentation(
                weighting_fun=self._params["graph_aggregation_weighting_fun"],
                graph_representation_size=self._params["graph_representation_size"],
                num_heads=self._params["graph_aggregation_num_heads"],
                scoring_mlp_layers=self._params["graph_aggregation_hidden_layers"],
                scoring_mlp_dropout_rate=self._params["graph_aggregation_dropout_rate"],
                transformation_mlp_layers=self._params["graph_aggregation_hidden_layers"],
                transformation_mlp_dropout_rate=self._params["graph_aggregation_dropout_rate"],
            )
            if self._property_type == PropertyTaskType.BINARY_CLASSIFICATION:
                self._property_predictor = MLPBinaryClassifierLayer(
                    mlp_layer_sizes=self._params["predictor_mlp_layer_sizes"],
                    dropout_rate=self._params["predictor_mlp_dropout_rate"],
                )
            elif self._property_type == PropertyTaskType.REGRESSION:
                self._property_predictor = MLPRegressionLayer(
                    mlp_layer_sizes=self._params["predictor_mlp_layer_sizes"],
                    dropout_rate=self._params["predictor_mlp_dropout_rate"],
                    property_stddev=property_stddev,
                )
            else:
                raise ValueError(f"Unknown property type {self._property_type}")

    def build(self, input_shapes: GraphPropertyPredictorInput):
        with tf.name_scope(self._name):
            self._graph_aggregation_layer.build(
                NodesToGraphRepresentationInput(
                    node_embeddings=input_shapes.node_representations,
                    node_to_graph_map=tf.TensorShape((None)),
                    num_graphs=tf.TensorShape(()),
                )
            )

            self._property_predictor.build(
                tf.TensorShape((None, self._params["graph_representation_size"]))
            )

            super().build(input_shapes)

    @tf.function(
        input_signature=(
            GraphPropertyPredictorInput(
                node_representations=tf.TensorSpec(
                    shape=tf.TensorShape((None, None)), dtype=tf.float32
                ),
                node_to_graph_map=tf.TensorSpec(shape=tf.TensorShape((None,)), dtype=tf.int32),
                num_graphs=tf.TensorSpec(shape=(), dtype=tf.int32),
                graph_ids_to_predict_for=tf.TensorSpec(shape=(None,), dtype=tf.int32),
            ),
            tf.TensorSpec(shape=(), dtype=tf.bool),
        )
    )
    def call(self, input: GraphPropertyPredictorInput, training: tf.Tensor) -> tf.Tensor:
        """
        Predict properties from graphs, given their node representations.

        Args:
            input: GraphPropertyPredictorInput holding the input representations.
            training: Flag indicating if we are training or not.

        Returns:
            List of property predictions, type float32.
        """
        graph_aggregation_input = NodesToGraphRepresentationInput(
            node_embeddings=input.node_representations,
            node_to_graph_map=input.node_to_graph_map,
            num_graphs=input.num_graphs,
        )

        graph_representations = self._graph_aggregation_layer(
            graph_aggregation_input, training=training
        )  # Shape [G, graph_representation_size]
        needed_graph_representations = tf.gather(
            # Note that the 1 * here is crucial to work around a bug in TensorFlow.
            # Concretely, tf.gather generates sparse gradients on params, which are not
            # supported as input for the auto-generated backward_* functions that TF
            # creates for @tf.function-annotated things.
            # Multiplication happens to transform the gradient into a dense representation,
            # which works around this problem.
            # See https://github.com/tensorflow/tensorflow/issues/36236 for the filed TF bug.
            params=1 * graph_representations,
            indices=input.graph_ids_to_predict_for,
        )
        return self._property_predictor(needed_graph_representations, training=training)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None), dtype=tf.float32),
            tf.TensorSpec(shape=(None), dtype=tf.float32),
        ]
    )
    def compute_task_metrics(
        self,
        property_predictions: tf.Tensor,
        property_labels: tf.Tensor,
    ) -> Tuple[tf.Tensor, Dict[str, Any]]:
        loss, batch_results = self._property_predictor.compute_task_metrics(
            predictions=property_predictions, labels=property_labels
        )
        batch_results["num_samples"] = tf.shape(property_labels)[0]
        return loss, batch_results

    def compute_epoch_metrics(self, task_results: List[Any]) -> Tuple[float, str]:
        total_num_samples = np.sum(
            batch_task_result["num_samples"] for batch_task_result in task_results
        )

        return self._property_predictor.compute_epoch_metrics(
            num_samples=tf.cast(total_num_samples, tf.float32), task_results=task_results
        )
