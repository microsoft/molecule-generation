import argparse
import os
import pickle
from typing import Dict

import numpy as np
from rdkit import Chem


from molecule_generation.utils.cli_utils import get_model_loading_parser
from molecule_generation.utils.moler_visualisation_utils import (
    GraphGenerationVisualiser,
    PropertyPredictionInformation,
    MoleculeGenerationAtomChoiceInfo,
    MoleculeGenerationEdgeChoiceInfo,
    MoleculeGenerationAttachmentPointChoiceInfo,
)


class TextGraphGenerationVisualiser(GraphGenerationVisualiser):
    def render_property_data(self, prop_infos: Dict[str, PropertyPredictionInformation]) -> None:
        print("= Molecule properties")
        for prop_name, prop_info in prop_infos.items():
            print(f" Property {prop_name:>15}:" f" Pred = {prop_info.prediction:7.3f}", end="")
            if prop_info.ground_truth is None:
                print()
            else:
                print(f" (True = {prop_info.ground_truth:7.3f})")

    def render_atom_data(
        self,
        atom_info: MoleculeGenerationAtomChoiceInfo,
        choice_descr: str,
        prob_threshold: float = 0.001,
    ) -> None:
        print(f"  - Atom/motif choices for {choice_descr}")
        if atom_info.true_type_idx is not None:
            correct_choices_str = ", ".join(
                [self.dataset._node_type_index_to_string[idx] for idx in atom_info.true_type_idx]
            )
            print(f"    Correct: {correct_choices_str}")

        for _, prob, descr in self.get_atom_and_motif_types_to_render(atom_info, prob_threshold):
            print(f"     {prob:.3f} - {descr}")

    def render_attachment_point_selection_step(
        self, step: int, attachment_point_info: MoleculeGenerationAttachmentPointChoiceInfo
    ) -> None:
        print(f" -- Step {step} - selecting attachment point")
        print("  - Attachment point choices")

        correct_choice = attachment_point_info.correct_attachment_point_idx

        for candidate, prob in zip(
            attachment_point_info.candidate_attachment_points,
            attachment_point_info.candidate_idx_to_prob,
        ):
            if correct_choice is None:
                correctness_info = ""
            else:
                correctness_info = f" {'[in' if not candidate == correct_choice else '  ['}correct]"

            print(f"    Node {candidate:2}: prob {prob:.3f} {correctness_info}")

    def render_molecule_gen_start(self, mol: Chem.Mol) -> None:
        print(f"= Steps to create {Chem.MolToSmiles(mol)}")

    def render_molecule_gen_edge_step(
        self, step: int, step_info: MoleculeGenerationEdgeChoiceInfo
    ) -> None:
        print(f" -- Step {step} - focusing on node {step_info.focus_node_idx}")
        print("  - Edge choices")
        for edge_info in step_info.candidate_edge_infos:
            edge_type_result_str = ", ".join(
                f"type {typ + 1} prob: {np.exp(logprob):.3f}"
                for typ, logprob in enumerate(edge_info.type_idx_to_logprobs)
            )
            if edge_info.correct is None:
                correctness_info = ""
            else:
                correctness_info = f" {'[in' if not edge_info.correct else '  ['}correct]"
            print(
                f"    Edge {step_info.focus_node_idx:2} -- {edge_info.target_node_idx:2}:   "
                f"prob {np.exp(edge_info.logprob):5.3f}, score {edge_info.score:7.3f}"
                f"{correctness_info} | {edge_type_result_str}"
            )

        if step_info.no_edge_correct is None:
            correctness_info = ""
        else:
            correctness_info = f" {'[in' if not step_info.no_edge_correct else '  ['}correct]"
        print(
            f"    No further edge: "
            f"prob {np.exp(step_info.no_edge_logprob):5.3f}, score {step_info.no_edge_score:7.3f}"
            + correctness_info
        )


def run_from_args(args) -> None:
    visualiser = TextGraphGenerationVisualiser(args.MODEL_DIR)
    for smiles_or_path in args.SMILES_OR_PATH:
        if os.path.exists(smiles_or_path):
            print(f">>> Molecule representations loaded from {smiles_or_path}")
            with open(smiles_or_path, "rb") as fh:
                samples = pickle.load(fh)
            visualiser.visualise_from_samples(samples)
        else:
            print(f">>> Molecule {smiles_or_path}")
            visualiser.visualise_from_smiles(smiles_or_path)


def get_argparser() -> argparse.ArgumentParser():
    parser = get_model_loading_parser(
        description="Visualise MoLeR molecule generation as text.", include_extra_args=False
    )
    parser.add_argument(
        "SMILES_OR_PATH",
        type=str,
        nargs="+",
        help="SMILES string(s) or paths of latent representations to visualise.",
    )
    return parser
