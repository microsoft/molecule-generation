from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Dict, NamedTuple, Optional

import tensorflow as tf
import numpy as np
from rdkit import Chem

from molecule_generation.models.cgvae import CGVAEOutput, GraphPropertyPredictorInput
from molecule_generation.utils.beam_utils import (
    MoleculeGenerationEdgeCandidateInfo,
    MoleculeGenerationStepInfo,
)
from molecule_generation.dataset.trace_dataset import DataFold
from molecule_generation.utils.model_utils import load_vae_model_and_dataset


class PropertyPredictionInformation(NamedTuple):
    prediction: float
    ground_truth: Optional[float]


class AtomPredictionInformation(NamedTuple):
    node_idx: int
    true_type_idx: Optional[int]
    type_idx_to_prob: List[float]


class GraphGenerationVisualiser(ABC):
    def __init__(self, trained_model_path: str):
        dataset, vae = load_vae_model_and_dataset(trained_model_path)
        self.dataset = dataset
        self.dataset.params[
            "trace_element_keep_prob"
        ] = 1.0  # Make sure that we keep all trace steps
        self.vae = vae
        self.supported_property_names = vae._graph_property_params.keys()

    @abstractmethod
    def render_property_data(self, prop_infos: Dict[str, PropertyPredictionInformation]) -> None:
        pass

    @abstractmethod
    def render_atom_data(self, atom_infos: List[AtomPredictionInformation]) -> None:
        pass

    @abstractmethod
    def render_molecule_gen_start(self, mol: Chem.Mol) -> None:
        pass

    @abstractmethod
    def render_molecule_gen_step(self, step: int, step_info: MoleculeGenerationStepInfo) -> None:
        pass

    def visualise_from_smiles(self, smiles: str):
        # First, load the raw sample and run the model on it
        mol, trace_sample = self.dataset.transform_smiles_to_sample(smiles)
        self.dataset._loaded_data[DataFold.TEST] = [trace_sample]
        tf_dataset = self.dataset.get_tensorflow_dataset(data_fold=DataFold.TEST)
        batch_features, batch_labels = next(iter(tf_dataset))
        predictions: CGVAEOutput = self.vae(batch_features, training=False)

        self.render_property_data(
            {
                prop_name: PropertyPredictionInformation(
                    prediction=predictions.predicted_properties[prop_name].numpy(),
                    ground_truth=trace_sample.graph_property_values[prop_name],
                )
                for prop_name in self.supported_property_names
            }
        )

        num_nodes = len(trace_sample.node_features)
        self.render_atom_data(
            [
                AtomPredictionInformation(
                    node_idx=node_idx,
                    true_type_idx=batch_labels["node_types"][node_idx].numpy(),
                    type_idx_to_prob=tf.nn.softmax(
                        predictions.node_classification_logits[node_idx, :]
                    ).numpy(),
                )
                for node_idx in range(num_nodes)
            ]
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
        for step, focus_node_idx in enumerate(batch_features["focus_nodes"].numpy()):
            focus_node_orig_idx = partial_node_to_orig_node_id[focus_node_idx]
            edge_choice_indices = focus_node_to_choice_indices[focus_node_idx]
            edge_choice_scores = [
                predictions.edge_logits[edge_choice_index].numpy()
                for edge_choice_index in edge_choice_indices
            ]
            # The special "no more edges" choices are appended to the end, so get that one:
            edge_choice_scores.append(
                predictions.edge_logits[total_num_valid_edge_choices + step].numpy()
            )
            edge_choice_scores = np.concatenate(edge_choice_scores, axis=0)
            edge_choice_logprobs = tf.nn.log_softmax(edge_choice_scores).numpy()

            candidate_edge_infos = []
            any_edge_candidate_correct = False
            for i, edge_choice_index in enumerate(edge_choice_indices):
                tgt_idx = batch_features["valid_edge_choices"][edge_choice_index, 1].numpy()
                tgt_node_orig_idx = partial_node_to_orig_node_id[tgt_idx]
                choice_correct = batch_labels["correct_edge_choices"][edge_choice_index].numpy() > 0
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

            self.render_molecule_gen_step(
                step,
                MoleculeGenerationStepInfo(
                    focus_node_idx=focus_node_orig_idx,
                    partial_molecule_adjacency_lists=trace_sample.partial_adjacency_lists[step],
                    candidate_edge_infos=candidate_edge_infos,
                    no_edge_score=edge_choice_scores[-1],
                    no_edge_logprob=edge_choice_logprobs[-1],
                    no_edge_correct=not any_edge_candidate_correct,
                ),
            )

    def visualise_from_samples(self, node_representations: np.ndarray):
        num_nodes = node_representations.shape[0]

        # First, run and render the property predictions:
        prop_prediction_input = GraphPropertyPredictorInput(
            node_representations=node_representations,
            node_to_graph_map=np.zeros(shape=(num_nodes,), dtype=np.int32),
            num_graphs=1,
            graph_ids_to_predict_for=[0],
        )
        property_data = {}
        for prop_name in self.supported_property_names:
            property_data[prop_name] = PropertyPredictionInformation(
                prediction=self.vae._property_predictors[prop_name](
                    prop_prediction_input, training=False
                ).numpy(),
                ground_truth=None,
            )
        self.render_property_data(property_data)

        # Second, run and render the atom type predictions:
        node_classification_logits = self.vae._node_to_label_layer(
            node_representations, training=False
        )
        self.render_atom_data(
            [
                AtomPredictionInformation(
                    node_idx=node_idx,
                    true_type_idx=None,
                    type_idx_to_prob=tf.nn.softmax(node_classification_logits[node_idx, :]).numpy(),
                )
                for node_idx in range(num_nodes)
            ]
        )
        # We also need to compute the picked node labels (which is just wrapping around the logits):
        node_labels = self.vae.classify_nodes(node_features=node_representations)

        # Finally, run the decoder while recording the trace, and then render that:
        beam_search_results = self.vae.decoder.beam_decode(
            node_types=node_labels,
            node_features=node_representations,
            beam_size=1,
            store_generation_traces=True,
        )

        final_molecule = beam_search_results[0].molecule
        self.render_molecule_gen_start(final_molecule)

        for step, step_info in enumerate(beam_search_results[0].generation_trace):
            self.render_molecule_gen_step(step, step_info)
