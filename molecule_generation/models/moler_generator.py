from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Any
from molecule_generation.models.moler_base_model import MoLeRBaseModel

import tensorflow as tf

from molecule_generation.layers.moler_decoder import MoLeRDecoderOutput


@dataclass
class MoLeRGeneratorOutput(MoLeRDecoderOutput):
    first_node_type_logits: tf.Tensor


class MoLeRGenerator(MoLeRBaseModel):
    """Uses the MoLeR decoder layer to generate a policy (in the reinforcement learning sense)
    for constructing molecules step-by-step.

    The states passed to the decoder are partially constructed molecules, and the decoder computes
    logits for extending these partial molecules in different ways.
    """

    def compute_final_node_representations(self, inputs, training: bool):
        return None

    def compute_task_output(
        self,
        batch_features: Dict[str, tf.Tensor],
        final_node_representations: Optional[
            Any
        ],  # Ignored: here for consistency of interface with moler_vae
        training: bool,
    ) -> MoLeRGeneratorOutput:
        molecule_representations = tf.zeros(
            (batch_features["num_partial_graphs_in_batch"], self.latent_dim)
        )
        decoder_output = self._get_decoder_output(
            batch_features=batch_features,
            molecule_representations=molecule_representations,
            training=training,
        )

        per_graph_molecule_representations = tf.zeros(
            (batch_features["num_graphs_in_batch"], self.latent_dim)
        )
        first_node_type_logits = self._decoder_layer.pick_first_node_type(
            input_molecule_representations=per_graph_molecule_representations,
            training=training,
        )

        return MoLeRGeneratorOutput(
            node_type_logits=decoder_output.node_type_logits,
            first_node_type_logits=first_node_type_logits,
            edge_candidate_logits=decoder_output.edge_candidate_logits,
            edge_type_logits=decoder_output.edge_type_logits,
            attachment_point_selection_logits=decoder_output.attachment_point_selection_logits,
        )

    def compute_task_metrics(
        self,
        batch_features: Dict[str, tf.Tensor],
        task_output: MoLeRGeneratorOutput,
        batch_labels: Dict[str, tf.Tensor],
    ) -> Dict[str, tf.Tensor]:
        total_loss, decoder_metrics = self._compute_decoder_loss_and_metrics(
            batch_features=batch_features, task_output=task_output, batch_labels=batch_labels
        )

        return {
            "loss": total_loss,
            "first_node_classification_loss": decoder_metrics.first_node_classification_loss,
            "edge_loss": decoder_metrics.edge_loss,
            "edge_type_loss": decoder_metrics.edge_type_loss,
            "attachment_point_selection_loss": decoder_metrics.attachment_point_selection_loss,
            "node_classification_loss": decoder_metrics.node_classification_loss,
        }

    def build(self, input_shapes: Dict[str, Any]):
        super().build(input_shapes=input_shapes)
        # Skip call to GraphTaskModel.build, since we don't want to build an
        # encoder GNN for this model
        self.built = True

    def compute_epoch_metrics(self, task_results: List[Any]) -> Tuple[float, str]:
        average_loss = self._dict_average(task_results, "loss")
        result_string = f"\n" f"Avg weighted sum. of graph losses: {average_loss: 7.4f}\n"

        graph_generation_losses = self._get_graph_generation_losses(task_results)
        result_string += self._format_graph_generation_losses(graph_generation_losses)

        return average_loss, result_string
