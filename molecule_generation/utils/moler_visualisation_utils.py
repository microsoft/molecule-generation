from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Dict, NamedTuple, Optional

import numpy as np
import tensorflow as tf
from rdkit import Chem

from molecule_generation.dataset.trace_dataset import DataFold
from molecule_generation.layers.moler_decoder import MoLeRDecoderState
from molecule_generation.layers.moler_decoder import (
    MoleculeGenerationEdgeCandidateInfo,
    MoleculeGenerationEdgeChoiceInfo,
    MoleculeGenerationAtomChoiceInfo,
    MoleculeGenerationAttachmentPointChoiceInfo,
)
from molecule_generation.models.moler_vae import MoLeRVaeOutput
from molecule_generation.utils.model_utils import load_vae_model_and_dataset
from molecule_generation.wrapper import ModelWrapper


class PropertyPredictionInformation(NamedTuple):
    prediction: float
    ground_truth: Optional[float]


class AtomPredictionInformation(NamedTuple):
    node_idx: int
    true_type_idx: Optional[List[int]]
    type_idx_to_prob: List[float]


class GraphGenerationVisualiser(ABC):
    def __init__(self, model_dir: str):
        dataset, vae = load_vae_model_and_dataset(ModelWrapper._get_model_file(model_dir))
        self.dataset = dataset
        self.dataset.params[
            "trace_element_keep_prob"
        ] = 1.0  # Make sure that we keep all trace steps
        self.dataset.params["max_nodes_per_batch"] = 5000
        self.dataset.params["max_partial_nodes_per_batch"] = 5000
        self.vae = vae
        self.supported_property_names = dataset.params["graph_properties"].keys()

    @abstractmethod
    def render_property_data(self, prop_infos: Dict[str, PropertyPredictionInformation]) -> None:
        pass

    @abstractmethod
    def render_atom_data(self, atom_infos: List[MoleculeGenerationAtomChoiceInfo]) -> None:
        pass

    @abstractmethod
    def render_molecule_gen_start(self, mol: Chem.Mol) -> None:
        pass

    @abstractmethod
    def render_molecule_gen_edge_step(
        self, step: int, step_info: MoleculeGenerationEdgeChoiceInfo
    ) -> None:
        pass

    @abstractmethod
    def render_attachment_point_selection_step(
        self, step: int, attachment_point_info: MoleculeGenerationAttachmentPointChoiceInfo
    ) -> None:
        pass

    def visualise_from_smiles(self, smiles: str):
        # First, load the raw sample and run the model on it
        mol, trace_sample = self.dataset.transform_smiles_to_sample(smiles)
        self.dataset._loaded_data[DataFold.TEST] = [trace_sample]
        tf_dataset = self.dataset.get_tensorflow_dataset(data_fold=DataFold.TEST)
        batch_features, batch_labels = next(iter(tf_dataset))
        predictions: MoLeRVaeOutput = self.vae(batch_features, training=False)

        self.render_property_data(
            {
                prop_name: PropertyPredictionInformation(
                    prediction=predictions.predicted_properties[prop_name].numpy(),
                    ground_truth=trace_sample.graph_property_values[prop_name],
                )
                for prop_name in self.supported_property_names
            }
        )

        # Now to the actual step-by-step thing. First, call a hook to allow doing pre-work:
        self.render_molecule_gen_start(mol)

        # Then, set up some translation maps we'll need:
        focus_node_to_choice_indices: Dict[int, List[int]] = defaultdict(list)
        for choice_idx, src_node_idx in enumerate(
            batch_features["valid_edge_choices"][:, 0].numpy()
        ):
            focus_node_to_choice_indices[src_node_idx].append(choice_idx)
        partial_node_to_orig_node_id = dict(
            enumerate(i.numpy() for i in batch_features["partial_node_to_original_node_map"])
        )

        total_num_valid_edge_choices = batch_features["valid_edge_choices"].shape[0]
        steps_requiring_node_choices = batch_features[
            "partial_graphs_requiring_node_choices"
        ].numpy()

        node_to_partial_graph = batch_features["node_to_partial_graph_map"].numpy()
        valid_attachment_point_choices = batch_features["valid_attachment_point_choices"].numpy()
        correct_attachment_point_choices = batch_labels["correct_attachment_point_choices"].numpy()

        graph_to_valid_attachment_point_choices = {}
        graph_to_correct_attachment_point_choice = {}

        num_nodes_per_partial_graph = defaultdict(int)

        for partial_graph_idx in node_to_partial_graph:
            num_nodes_per_partial_graph[partial_graph_idx] += 1

        for node_idx in valid_attachment_point_choices:
            graph_idx = node_to_partial_graph[node_idx]

            if graph_idx not in graph_to_valid_attachment_point_choices:
                graph_to_valid_attachment_point_choices[graph_idx] = []

            graph_to_valid_attachment_point_choices[graph_idx].append(
                partial_node_to_orig_node_id[node_idx]
            )

        for node_idx in correct_attachment_point_choices:
            node_idx = valid_attachment_point_choices[node_idx]

            graph_idx = node_to_partial_graph[node_idx]
            graph_to_correct_attachment_point_choice[graph_idx] = partial_node_to_orig_node_id[
                node_idx
            ]

        for step, focus_node_idx in enumerate(batch_features["focus_nodes"].numpy()):
            focus_node_orig_idx = partial_node_to_orig_node_id[focus_node_idx]

            if step in graph_to_valid_attachment_point_choices:
                indices = list(
                    np.where(node_to_partial_graph[valid_attachment_point_choices] == step)[0]
                )
                logits = predictions.attachment_point_selection_logits.numpy()[indices]

                num_nodes_in_current_graph = num_nodes_per_partial_graph[step]
                num_nodes_in_previous_graph = (
                    num_nodes_per_partial_graph[step - 1] if step > 0 else 0
                )

                added_motif_nodes = list(
                    range(num_nodes_in_previous_graph, num_nodes_in_current_graph)
                )

                self.render_attachment_point_selection_step(
                    step,
                    MoleculeGenerationAttachmentPointChoiceInfo(
                        partial_molecule_adjacency_lists=trace_sample.partial_adjacency_lists[step],
                        motif_nodes=added_motif_nodes,
                        candidate_attachment_points=graph_to_valid_attachment_point_choices[step],
                        candidate_idx_to_prob=tf.nn.softmax(logits),
                        correct_attachment_point_idx=graph_to_correct_attachment_point_choice[step],
                    ),
                )
            else:
                edge_choice_indices = focus_node_to_choice_indices[focus_node_idx]
                edge_choice_scores = [
                    predictions.edge_candidate_logits[edge_choice_index].numpy()
                    for edge_choice_index in edge_choice_indices
                ]
                # The special "no more edges" choices are appended to the end, so get that one:
                edge_choice_scores.append(
                    predictions.edge_candidate_logits[total_num_valid_edge_choices + step].numpy()
                )
                edge_choice_scores = np.array(edge_choice_scores)
                edge_choice_logprobs = tf.nn.log_softmax(edge_choice_scores).numpy()

                candidate_edge_infos = []
                any_edge_candidate_correct = False
                for i, edge_choice_index in enumerate(edge_choice_indices):
                    tgt_idx = batch_features["valid_edge_choices"][edge_choice_index, 1].numpy()
                    tgt_node_orig_idx = partial_node_to_orig_node_id[tgt_idx]
                    choice_correct = (
                        batch_labels["correct_edge_choices"][edge_choice_index].numpy() > 0
                    )
                    any_edge_candidate_correct |= choice_correct
                    type_logprobs = tf.nn.log_softmax(
                        predictions.edge_type_logits[edge_choice_index, :]
                    )
                    candidate_edge_infos.append(
                        MoleculeGenerationEdgeCandidateInfo(
                            target_node_idx=tgt_node_orig_idx,
                            score=edge_choice_scores[i],
                            logprob=edge_choice_logprobs[i],
                            correct=choice_correct,
                            type_idx_to_logprobs=type_logprobs.numpy(),
                        )
                    )

                self.render_molecule_gen_edge_step(
                    step,
                    MoleculeGenerationEdgeChoiceInfo(
                        focus_node_idx=focus_node_orig_idx,
                        partial_molecule_adjacency_lists=trace_sample.partial_adjacency_lists[step],
                        candidate_edge_infos=candidate_edge_infos,
                        no_edge_score=edge_choice_scores[-1],
                        no_edge_logprob=edge_choice_logprobs[-1],
                        no_edge_correct=not any_edge_candidate_correct,
                    ),
                )

            if step in steps_requiring_node_choices:
                node_choice_idx = np.where(steps_requiring_node_choices == step)[0][0]
                one_hot_labels = batch_labels["correct_node_type_choices"][node_choice_idx].numpy()
                true_type_idx = one_hot_labels.nonzero()[0]
                self.render_atom_data(
                    [
                        MoleculeGenerationAtomChoiceInfo(
                            node_idx=focus_node_orig_idx + 1,
                            true_type_idx=true_type_idx,
                            type_idx_to_prob=tf.nn.softmax(
                                predictions.node_type_logits[node_choice_idx, :]
                            ).numpy(),
                        )
                    ]
                )

    def visualise_from_samples(self, molecule_representation: np.ndarray):

        property_data: Dict[str, PropertyPredictionInformation] = {}

        for prop_name in self.supported_property_names:
            property_data[prop_name] = PropertyPredictionInformation(
                prediction=self.vae._property_predictors[prop_name](
                    [molecule_representation], training=False
                ).numpy(),
                ground_truth=None,
            )
        self.render_property_data(property_data)

        # Finally, run the decoder while recording the trace, and then render that:
        decoder_states: MoLeRDecoderState = self.vae.decoder.decode(
            graph_representations=[molecule_representation],
            store_generation_traces=True,
        )[0]

        # Now to the actual step-by-step thing. First, call a hook to allow doing pre-work:
        mol = decoder_states.molecule
        self.render_molecule_gen_start(mol)

        num_atom_selection_steps = len(decoder_states.atom_selection_steps)

        assert len(decoder_states.edge_selection_steps) == num_atom_selection_steps
        assert len(decoder_states.attachment_point_selection_steps) == num_atom_selection_steps

        step = 1
        for atom, edge, attachment_point in zip(
            decoder_states.atom_selection_steps,
            decoder_states.edge_selection_steps,
            decoder_states.attachment_point_selection_steps,
        ):
            if attachment_point is not None:
                self.render_attachment_point_selection_step(step, attachment_point)
                step += 1
            elif edge is not None:
                self.render_molecule_gen_edge_step(
                    step,
                    edge,
                )
                step += 1

            if atom is not None:
                self.render_atom_data([atom])