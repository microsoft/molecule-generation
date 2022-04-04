from typing import Optional, Dict, Tuple, List, Any
from collections import defaultdict, deque
import time

import tensorflow as tf
import numpy as np


class EpochMetricsLogger:
    """Logs metrics for an epoch of training"""

    def __init__(
        self, *, window_size: int = 100, quiet: bool, aml_run: Optional, training: bool
    ) -> None:

        self._window_size = window_size
        self._quiet = quiet
        self._aml_run = aml_run
        self._training = training

        # Initialise everything in case you don't want to use this as a contextmanager
        self._start_logging()

    def _start_logging(self):
        """Initialise counters, timer, buffers"""
        # This will hold the latest window_size values of each metric
        self._raw_metrics = defaultdict(lambda: deque(maxlen=self._window_size))

        # This will hold mean of the last window_size values of each metric
        self._moving_average_metrics = None

        self._total_loss = 0.0
        self._total_num_graphs = 0
        self._task_results = []
        self._step = 0

        self._start_time = time.time()

        # This will be populated when exiting EpochMetricsLogger context
        self._total_time = None
        self._finished = False

    def log_step_metrics(self, task_metrics, batch_features):
        """Log metrics for a single step"""
        assert not self._finished
        self._step += 1
        self._total_loss += float(task_metrics["loss"])
        self._total_num_graphs += int(batch_features["num_graphs_in_batch"])
        self._task_results.append(task_metrics)
        self._append_loss_metrics(
            task_metrics,
            int(batch_features["num_graphs_in_batch"]),
            int(batch_features["num_partial_graphs_in_batch"]),
        )
        if self._step >= self._window_size and self._step % self._window_size == 0:
            self._moving_average_metrics = self._get_moving_average_metrics()
            if self._aml_run is not None:
                for k, v in self._moving_average_metrics.items():
                    self._aml_run.log("smoothed_" + k, float(v))

        # Tensorboard logging:
        batch_graph_average_loss = task_metrics["loss"] / float(
            batch_features["num_graphs_in_batch"]
        )
        if self._training:
            s = tf.summary.experimental.get_step()
            tf.summary.experimental.set_step(s + 1)
            tf.summary.scalar(
                "batch_graph_av_loss",
                data=batch_graph_average_loss,
            )

        # Text logging.
        if not self._quiet:
            epoch_graph_average_loss = self._total_loss / float(self._total_num_graphs)
            steps_per_second = self._step / (time.time() - self._start_time)
            print_string = (
                f"   Step: {self._step:4d}"
                f"  |  Epoch graph avg. loss = {epoch_graph_average_loss:.5f}"
                f"  |  Batch graph avg. loss = {batch_graph_average_loss:.5f}"
            )
            if self._moving_average_metrics is not None:
                mean_num_graphs = np.mean(self._raw_metrics["num_input_graphs"])
                mean_num_partial_graphs = np.mean(self._raw_metrics["num_partial_graphs_in_batch"])
                print_string += (
                    f"  |  Moving avg. loss = {self._moving_average_metrics['loss']:.5f}"
                    f" , avg #graphs = {mean_num_graphs:.2f}"
                    f" , avg #trace steps = {mean_num_partial_graphs:.2f}"
                )
            print_string += f"  |  Steps per sec = {steps_per_second:.5f}"
            print(print_string, end="\r")

    def __enter__(self):
        self._start_logging()
        return self

    def __exit__(self, *exc):
        """Stop logging, stop timer"""
        if not self._quiet:
            print("\r\x1b[K", end="")
        self._total_time = time.time() - self._start_time
        self._finished = True
        return False

    def _append_loss_metrics(
        self,
        loss_dict: Dict[str, tf.Tensor],
        num_graphs_in_batch: int,
        num_partial_graphs_in_batch: int,
    ):
        self._raw_metrics["num_input_graphs"].append(num_graphs_in_batch)
        self._raw_metrics["num_partial_graphs_in_batch"].append(num_partial_graphs_in_batch)
        for k, v in loss_dict.items():
            # Metrics that are not tensors will be skipped
            if not isinstance(v, tf.Tensor):
                continue
            self._raw_metrics[k].append(v.numpy())

    def _get_moving_average_metrics(self) -> Dict[str, float]:
        """Get metrics averaged over the last window_size steps"""

        # Compute the total number of graphs, but guard against division by 0.
        total_num_input_graphs = max(1, sum(self._raw_metrics["num_input_graphs"]))
        total_metrics = {k: sum(v) for k, v in self._raw_metrics.items()}

        average_metrics = {
            k: v / total_num_input_graphs
            for k, v in total_metrics.items()
            if k not in ["num_input_graphs", "num_partial_graphs_in_batch"]
        }

        # Guard against division by 0
        num_batches = max(1, len(self._raw_metrics["num_input_graphs"]))

        # Add metrics that count input / partial graphs, but name them descriptively.
        average_metrics["num_input_graphs_per_batch"] = (
            total_metrics["num_input_graphs"] / num_batches
        )
        average_metrics["num_partial_graphs_per_batch"] = (
            total_metrics["num_partial_graphs_in_batch"] / num_batches
        )
        average_metrics["num_partial_graphs_per_input_graph"] = (
            total_metrics["num_partial_graphs_in_batch"] / total_num_input_graphs
        )

        return average_metrics

    def get_epoch_summary(self) -> Tuple[float, float, List[Any]]:
        """
        Returns (mean loss per graph, graphs processed per second, metrics for each step)
        """
        # Not to be called before exiting context, since that's when the _total_time is populated
        assert self._finished
        return (
            self._total_loss / float(self._total_num_graphs),
            float(self._total_num_graphs) / self._total_time,
            self._task_results,
        )
