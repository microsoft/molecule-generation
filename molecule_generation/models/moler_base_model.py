from typing import Dict, Optional, Any, Iterable, Tuple, List

import tensorflow as tf
from tf2_gnn import GraphTaskModel
import numpy as np

from molecule_generation.utils.training_utils import get_class_balancing_weights
from molecule_generation.layers.moler_decoder import (
    MoLeRDecoder,
    MoLeRDecoderInput,
    MoLeRDecoderMetrics,
    MoLeRDecoderOutput,
)
from molecule_generation.utils.epoch_metrics_logger import EpochMetricsLogger
from molecule_generation.dataset.trace_dataset import TraceDataset


class MoLeRBaseModel(GraphTaskModel):
    """Common base class for both MoLeRGenerator and MoLeRVae"""

    __decoder_prefix = "decoder_"

    @classmethod
    def decoder_prefix(cls) -> str:
        return cls.__decoder_prefix

    @classmethod
    def get_default_hyperparameters(cls, mp_style: Optional[str] = None) -> Dict[str, Any]:
        base_hypers = super().get_default_hyperparameters(mp_style)
        base_hypers.update(
            {
                # Dimension of the context/conditioning vector supplied to the MoLeRDecoder.
                # MoLeRGenerator will set this conditioning vector to all-zeros.
                "latent_repr_size": 512,
                # Relative contributions of the different next step prediction losses:
                "node_classification_loss_weight": 1.0,
                "first_node_classification_loss_weight": 0.07,
                "edge_selection_loss_weight": 1.0,
                "edge_type_loss_weight": 1.0,
                "attachment_point_selection_weight": 1.0,
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

        self._latent_repr_dim = params["latent_repr_size"]

        # Keep track of the training step as a TF variable
        self._train_step_counter = tf.Variable(
            initial_value=0, trainable=False, dtype=tf.int32, name="training_step"
        )

        # Get some information out from the dataset:
        next_node_type_distribution = dataset.metadata.get("train_next_node_type_distribution")

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
    def uses_motifs(self) -> bool:
        return self._uses_motifs

    @property
    def uses_categorical_features(self) -> bool:
        return self._node_categorical_num_classes is not None

    @property
    def decoder(self):
        return self._decoder_layer

    @property
    def latent_dim(self):
        return self._latent_repr_dim

    def build(self, input_shapes: Dict[str, Any]):
        """Build decoder."""
        latent_graph_representation_shape = tf.TensorShape((None, self.latent_dim))

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

        # Skip call to GraphTaskModel.build, in case we don't want to build an
        # encoder GNN.

    def _compute_decoder_loss_and_metrics(
        self, batch_features, task_output, batch_labels
    ) -> Tuple[tf.Tensor, MoLeRDecoderMetrics]:
        decoder_metrics = self.decoder.compute_metrics(
            batch_features=batch_features, batch_labels=batch_labels, task_output=task_output
        )

        total_loss = (
            self._params["node_classification_loss_weight"]
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

        return total_loss, decoder_metrics

    def _get_decoder_output(
        self, *, batch_features, molecule_representations, training
    ) -> MoLeRDecoderOutput:
        partial_adjacency_lists: Tuple[tf.Tensor, ...] = tuple(
            batch_features[f"partial_adjacency_list_{edge_type_idx}"]
            for edge_type_idx in range(self._num_edge_types)
        )  # Each element has shape (E, 2)
        decoder_output = self._decoder_layer(
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
        )
        return decoder_output

    def run_on_data_iterator(
        self,
        data_iterator: Iterable[Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]],
        quiet: bool = False,
        training: bool = True,
        max_num_steps: Optional[int] = None,  # Run until dataset ends if None
        aml_run: Optional[Any] = None,
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

    @staticmethod
    def _dict_average(task_results, key):
        return np.average([r[key] for r in task_results])

    def _get_graph_generation_losses(self, task_results: List[Any]) -> List[Tuple[str, str]]:
        """Average the reconstruction losses recorded in task_results"""

        average_node_classification_loss = self._dict_average(
            task_results, "node_classification_loss"
        )
        average_first_node_classification_loss = self._dict_average(
            task_results, "first_node_classification_loss"
        )
        average_edge_loss = self._dict_average(task_results, "edge_loss")
        average_edge_type_loss = self._dict_average(task_results, "edge_type_loss")
        if self.uses_motifs:
            average_attachment_point_selection_loss = self._dict_average(
                task_results, "attachment_point_selection_loss"
            )
        else:
            average_attachment_point_selection_loss = None
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

        return graph_generation_losses

    @staticmethod
    def _format_graph_generation_losses(graph_generation_losses: List[Tuple[str, str]]) -> str:
        result_string = ""
        name_column_width = max(len(prefix) for (prefix, _) in graph_generation_losses) + 1

        for prefix, loss in graph_generation_losses:
            # Use extra spaces to align all graph generation losses.
            result_string += prefix.ljust(name_column_width) + loss

        return result_string
