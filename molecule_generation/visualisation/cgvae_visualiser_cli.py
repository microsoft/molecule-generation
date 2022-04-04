#!/usr/bin/env python3
import os
import pickle
import warnings
from typing import List, Dict

import numpy as np
import tensorflow as tf
from dpu_utils.utils import run_and_debug
from rdkit import Chem


from molecule_generation.utils.cli_utils import get_model_loading_parser
from molecule_generation.utils.cgvae_visualisation_utils import (
    GraphGenerationVisualiser,
    PropertyPredictionInformation,
    AtomPredictionInformation,
    MoleculeGenerationStepInfo,
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

    def render_atom_data(self, atom_infos: List[AtomPredictionInformation]) -> None:
        print("= Atom types")
        num_atom_types = len(self.dataset._node_type_index_to_string)
        for atom_info in atom_infos:
            node_type_strs = [
                f"{self.dataset._node_type_index_to_string[node_typ_idx]}:"
                f" {atom_info.type_idx_to_prob[node_typ_idx]:.3f}"
                # Skip the first type, "UNK":
                for node_typ_idx in range(1, num_atom_types)
            ]
            if atom_info.true_type_idx is not None:
                true_node_type_info = (
                    f" (True: {self.dataset._node_type_index_to_string[atom_info.true_type_idx]})"
                )
            else:
                true_node_type_info = ""

            print(
                f" Node {atom_info.node_idx:2}{true_node_type_info}:\t{', '.join(node_type_strs)}"
            )

    def render_molecule_gen_start(self, mol: Chem.Mol) -> None:
        print(f"= Edge Steps to create {Chem.MolToSmiles(mol)}")

    def render_molecule_gen_step(self, step: int, step_info: MoleculeGenerationStepInfo) -> None:
        print(f" -- Step {step} - focusing on node {step_info.focus_node_idx}")
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
                f"   Edge {step_info.focus_node_idx:2} -- {edge_info.target_node_idx:2}:   "
                f"prob {np.exp(edge_info.logprob):5.3f}, score {edge_info.score:7.3f}"
                f"{correctness_info} | {edge_type_result_str}"
            )

        if step_info.no_edge_correct is None:
            correctness_info = ""
        else:
            correctness_info = f" {'[in' if not step_info.no_edge_correct else '  ['}correct]"
        print(
            f"   No further edge: "
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


def run():
    parser = get_model_loading_parser(
        description="Visualise CGVAE molecule generation as text.", include_extra_args=False
    )
    parser.add_argument(
        "SMILES_OR_PATH",
        type=str,
        nargs="+",
        help="SMILES string(s) or paths of latent representations to visualise.",
    )
    parser.add_argument("--debug", dest="debug", action="store_true", help="Enable debug routines")
    args = parser.parse_args()

    # Shut up tensorflow:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    tf.get_logger().setLevel("ERROR")
    warnings.simplefilter("ignore")

    run_and_debug(lambda: run_from_args(args), args.debug)


if __name__ == "__main__":
    run()
