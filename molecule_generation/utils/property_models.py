"""Task for (scalar) property prediction, starting from a vector representation."""
import enum
from abc import abstractmethod
from typing import Any, Dict, List, Tuple, Callable, Optional

import numpy as np
import tensorflow as tf
import sklearn.metrics as metrics
from dpu_utils.tf2utils import MLP


class PropertyTaskType(enum.Enum):
    BINARY_CLASSIFICATION = enum.auto()
    REGRESSION = enum.auto()


class PropertyPredictionLayer(tf.keras.layers.Layer):
    @abstractmethod
    def compute_task_metrics(
        self, predictions: tf.Tensor, labels: tf.Tensor
    ) -> Tuple[tf.Tensor, Dict[str, Any]]:
        """Given the output of `call`, compute loss and potential additional metrics.

        Args:
            predictions: Tensor of shape [B, T], where B is the batch dimension and
                T the number of outputs produced by `call` per example.
            labels: Tensor of shape [B, L], where B is the batch dimension and
                L is the dimensionality of the labels.

        Returns:
            Pair of loss (a scalar value) and a dictionary with additional results
            for this minibatch.
        """
        pass

    @abstractmethod
    def compute_epoch_metrics(
        self, num_samples: tf.float32, task_results: List[Dict[str, Any]]
    ) -> Tuple[float, str]:
        """Given the outputs of `compute_task_metrics` for all batches of an epoch,
        compute an epoch-level metric and a string description of current results.

        Args:
            num_samples: Number of samples in the entire epoch.
            task_results: List of outputs collected from repeated calls to
                compute_task_metrics`.

        Returns:
            Pair of a metric (a scalar value) to minimize and a string description
            of current results.
        """
        pass

    @staticmethod
    @abstractmethod
    def print_evaluation_report(
        prop_name: str, predictions, labels, log_fun: Callable[[str], None] = print
    ) -> None:
        """Print an evaluation report for the results of this property prediction layer,
        given all computed predictions and labels.

        Args:
            prop_name: Name of property this layer is predicting, for readability.
            predictions: Tensor of shape [D, T], where D is the number of samples in the
                dataset and T the number of outputs produced by `call` per example.
            labels: Tensor of shape [D, L], where  D is the number of samples in the
                dataset and L is the dimensionality of the labels.
            log_fun: Optional function used to print the report.
        """
        pass


class MLPRegressionLayer(PropertyPredictionLayer):
    def __init__(
        self,
        mlp_layer_sizes: List[int],
        dropout_rate: float,
        property_stddev: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._mlp_layer_sizes = mlp_layer_sizes
        self._dropout_rate = dropout_rate
        if property_stddev == 0:
            raise ValueError(
                f"Standard deviation of 0.0 cannot be used for normalisation - please turn off normalisation."
            )
        self._property_stddev = property_stddev
        self._mlp = MLP(out_size=1, hidden_layers=mlp_layer_sizes, dropout_rate=dropout_rate)

    def build(self, input_shape):
        self._mlp.build(input_shape)
        super().build(input_shape)

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.bool),
        )
    )
    def call(self, input, training: bool = False):
        return tf.squeeze(self._mlp(input, training=training))

    def compute_task_metrics(
        self, predictions: tf.Tensor, labels: tf.Tensor
    ) -> Tuple[tf.Tensor, Dict[str, Any]]:
        abs_error = tf.abs(labels - predictions)
        squared_error = tf.square(abs_error)

        metrics = {"absolute_error": abs_error, "squared_error": squared_error}

        # Loss is batch-mean squared error (following Gilmer et al 2017); but we support normalising
        # values as well, so that losses for different properties are on the same scale:
        if self._property_stddev is None:
            loss = tf.reduce_mean(squared_error)
        else:
            normalised_abs_error = abs_error / self._property_stddev
            normalised_squared_error = tf.square(normalised_abs_error)
            metrics["normalised_absolute_error"] = normalised_abs_error
            metrics["normalised_squared_error"] = normalised_squared_error
            loss = tf.reduce_mean(normalised_squared_error)

        return (loss, metrics)

    def compute_epoch_metrics(
        self, num_samples: tf.float32, task_results: List[Dict[str, Any]]
    ) -> Tuple[float, str]:
        epoch_abs_err = np.sum(
            np.sum(batch_task_result["absolute_error"]) for batch_task_result in task_results
        )
        epoch_squared_err = np.sum(
            np.sum(batch_task_result["squared_error"]) for batch_task_result in task_results
        )
        num_samples = num_samples.numpy()
        epoch_mae = epoch_abs_err / num_samples
        epoch_mse = epoch_squared_err / num_samples
        result_str = f"MAE {epoch_mae:.2f}, MSE {epoch_mse:.2f}"

        # Metric to optimise is epoch-mean absolute error (following Gilmer et al 2017); but we
        # support normalising values as well, so that metrics for different properties are on
        # the same scale:
        if self._property_stddev is None:
            metric_to_optimise = epoch_mae
        else:
            epoch_normalised_abs_err = np.sum(
                np.sum(batch_task_result["normalised_absolute_error"])
                for batch_task_result in task_results
            )
            epoch_normalised_mae = epoch_normalised_abs_err / num_samples
            metric_to_optimise = epoch_normalised_mae
            result_str = f"{result_str} (norm MAE: {metric_to_optimise:.2f})"

        return metric_to_optimise, result_str

    @staticmethod
    def print_evaluation_report(
        prop_name: str, predictions, labels, log_fun: Callable[[str], None] = print
    ) -> None:
        mae = metrics.mean_absolute_error(y_true=labels, y_pred=predictions)
        mse = metrics.mean_squared_error(y_true=labels, y_pred=predictions)
        max_err = metrics.max_error(y_true=labels, y_pred=predictions)
        expl_var = metrics.explained_variance_score(y_true=labels, y_pred=predictions)
        r2_score = metrics.r2_score(y_true=labels, y_pred=predictions)

        log_fun(f"Property {prop_name}:")
        log_fun(f" Mean Absolute Error: {mae:.3f}")
        log_fun(f" Mean Squared Error:  {mse:.3f}")
        log_fun(f" Maximum Error:       {max_err:.3f}")
        log_fun(f" Explained Variance:  {expl_var:.3f}")
        log_fun(f" R2 Score:            {r2_score:.3f}")

    @staticmethod
    def log_evaluation_report(
        prop_name: str, predictions, labels, aml_run=None, log_fun: Callable[[str], None] = print
    ) -> None:
        mae = metrics.mean_absolute_error(y_true=labels, y_pred=predictions)
        mse = metrics.mean_squared_error(y_true=labels, y_pred=predictions)
        max_err = metrics.max_error(y_true=labels, y_pred=predictions)
        expl_var = metrics.explained_variance_score(y_true=labels, y_pred=predictions)
        r2_score = metrics.r2_score(y_true=labels, y_pred=predictions)

        log_fun(f"Property {prop_name}:")
        log_fun(f" Mean Absolute Error: {mae:.3f}")
        log_fun(f" Mean Squared Error:  {mse:.3f}")
        log_fun(f" Maximum Error:       {max_err:.3f}")
        log_fun(f" Explained Variance:  {expl_var:.3f}")
        log_fun(f" R2 Score:            {r2_score:.3f}")

        if aml_run:
            aml_run.log_row(
                f"{prop_name}_test_metrics",
                mean_abs_err=float(mae),
                mse=float(mse),
                max_err=float(max_err),
                explained_variance=float(expl_var),
                r2_score=float(r2_score),
            )


class MLPBinaryClassifierLayer(MLPRegressionLayer):
    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.bool),
        )
    )
    def call(self, input, training: bool = False):
        return tf.nn.sigmoid(super().call(input, training=training))

    def compute_task_metrics(
        self, predictions: tf.Tensor, labels: tf.Tensor
    ) -> Tuple[float, Dict[str, Any]]:
        ce = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(
                y_true=labels, y_pred=predictions, from_logits=False
            )
        )
        num_correct = tf.reduce_sum(
            tf.cast(tf.math.equal(labels, tf.math.round(predictions)), tf.int32)
        )

        return ce, {"num_correct": num_correct}

    def compute_epoch_metrics(
        self, num_samples: tf.float32, task_results: List[Dict[str, Any]]
    ) -> Tuple[float, str]:
        epoch_num_correct = np.sum(
            batch_task_result["num_correct"] for batch_task_result in task_results
        )
        epoch_acc = tf.cast(epoch_num_correct, tf.float32) / num_samples
        result_str = f"acc {epoch_acc.numpy():.2%}"

        return -epoch_acc.numpy(), result_str

    @staticmethod
    def print_evaluation_report(
        prop_name: str, predictions, labels, log_fun: Callable[[str], None] = print
    ) -> None:
        rounded_predictions = np.round(predictions)
        acc = metrics.accuracy_score(y_true=labels, y_pred=rounded_predictions)
        balanced_acc = metrics.balanced_accuracy_score(y_true=labels, y_pred=rounded_predictions)
        precicision = metrics.precision_score(y_true=labels, y_pred=rounded_predictions)
        recall = metrics.recall_score(y_true=labels, y_pred=rounded_predictions)
        f1_score = metrics.f1_score(y_true=labels, y_pred=rounded_predictions)
        roc_auc = metrics.roc_auc_score(y_true=labels, y_score=predictions)

        log_fun(f"Property {prop_name}:")
        log_fun(f" Accuracy:          {acc:.2%}")
        log_fun(f" Balanced Accuracy: {balanced_acc:.4f}")
        log_fun(f" Precision:         {precicision:.4f}")
        log_fun(f" Recall:            {recall:.4f}")
        log_fun(f" F1 Score:          {f1_score:.4f}")
        log_fun(f" ROC AUC:           {roc_auc:.4f}")

    @staticmethod
    def log_evaluation_report(
        prop_name: str, predictions, labels, aml_run=None, log_fun: Callable[[str], None] = print
    ) -> None:
        rounded_predictions = np.round(predictions)
        acc = metrics.accuracy_score(y_true=labels, y_pred=rounded_predictions)
        balanced_acc = metrics.balanced_accuracy_score(y_true=labels, y_pred=rounded_predictions)
        precision = metrics.precision_score(y_true=labels, y_pred=rounded_predictions)
        recall = metrics.recall_score(y_true=labels, y_pred=rounded_predictions)
        f1_score = metrics.f1_score(y_true=labels, y_pred=rounded_predictions)
        roc_auc = metrics.roc_auc_score(y_true=labels, y_score=predictions)

        log_fun(f"Property {prop_name}:")
        log_fun(f" Accuracy:          {acc:.2%}")
        log_fun(f" Balanced Accuracy: {balanced_acc:.4f}")
        log_fun(f" Precision:         {precision:.4f}")
        log_fun(f" Recall:            {recall:.4f}")
        log_fun(f" F1 Score:          {f1_score:.4f}")
        log_fun(f" ROC AUC:           {roc_auc:.4f}")

        if aml_run:
            aml_run.log_row(
                f"{prop_name}_test_metrics",
                accuracy=float(acc),
                balanced_accuracy=float(balanced_acc),
                precision=float(precision),
                recall=float(recall),
                fl_score=float(f1_score),
                roc_auc_score=float(roc_auc),
            )
