#!/usr/bin/env python3
import os
import pickle
import warnings
from typing import List, Dict, Tuple

import numpy as np
import tensorflow as tf
from dpu_utils.utils import run_and_debug
from rdkit import Chem
from rdkit.Chem import BondType, RWMol, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry.rdGeometry import Point2D

from molecule_generation.utils.cli_utils import get_model_loading_parser
from molecule_generation.utils.cgvae_visualisation_utils import (
    GraphGenerationVisualiser,
    PropertyPredictionInformation,
    AtomPredictionInformation,
    MoleculeGenerationStepInfo,
)


EDGE_TYPE_IDX_TO_BOND_TYPE = {
    0: BondType.SINGLE,
    1: BondType.DOUBLE,
    2: BondType.TRIPLE,
}


class HTMLGraphGenerationVisualiser(GraphGenerationVisualiser):
    def render_property_data(self, prop_infos: Dict[str, PropertyPredictionInformation]) -> None:
        print(f"<h2>Molecule Properties</h2>", file=self.__out_fh)
        print(f"<table>", file=self.__out_fh)
        print(f" <tr>", file=self.__out_fh)
        print(
            f"  <th>Property</th>" f"  <th>Prediction</th>" f"  <th>True value</th>",
            file=self.__out_fh,
        )
        print(f" </tr>", file=self.__out_fh)
        for prop_name, prop_info in prop_infos.items():
            print(f" <tr>", file=self.__out_fh)
            print(
                f"  <td>{prop_name}</td>" f"  <td>{prop_info.prediction:7.3f}</td>",
                file=self.__out_fh,
                end="",
            )
            if prop_info.ground_truth is None:
                print("  <td>-</td>", file=self.__out_fh)
            else:
                print(f"  <td>{prop_info.ground_truth:7.3f}</td>", file=self.__out_fh)
            print(f" </tr>", file=self.__out_fh)
        print(f"</table>", file=self.__out_fh)

    def render_atom_data(self, atom_infos: List[AtomPredictionInformation]) -> None:
        print(f"<h2>Atom Types</h2>", file=self.__out_fh)
        print(f"<table>", file=self.__out_fh)
        print(f" <tr>", file=self.__out_fh)
        print(f"  <th>Node Id</th>", file=self.__out_fh)
        print(f"  <th>True Type</th>", file=self.__out_fh)
        num_atom_types = len(self.dataset._node_type_index_to_string)
        # Skip the first type, "UNK":
        for atom_type_idx in range(1, num_atom_types):
            print(
                f"  <th>{self.dataset._node_type_index_to_string[atom_type_idx]}</th>",
                file=self.__out_fh,
            )
        print(f" </tr>", file=self.__out_fh)

        for atom_info in atom_infos:
            print(f" <tr>", file=self.__out_fh)
            print(f"  <td>{atom_info.node_idx}</td>", file=self.__out_fh)
            if atom_info.true_type_idx is not None:
                print(
                    f"  <td><b>{self.dataset._node_type_index_to_string[atom_info.true_type_idx]}</b></td>",
                    file=self.__out_fh,
                )
            else:
                print(
                    f"  <td>n/a</td>",
                    file=self.__out_fh,
                )
            max_prob_atom_idx = np.argmax(atom_info.type_idx_to_prob)
            for atom_type_idx in range(1, num_atom_types):
                if atom_type_idx == max_prob_atom_idx:
                    print(
                        f"  <td><b>{atom_info.type_idx_to_prob[atom_type_idx]:.3f}</b></td>",
                        file=self.__out_fh,
                    )
                else:
                    print(
                        f"  <td>{atom_info.type_idx_to_prob[atom_type_idx]:.3f}</td>",
                        file=self.__out_fh,
                    )
            print(f" </tr>", file=self.__out_fh)
        print(f"</table>", file=self.__out_fh)

    def render_molecule_gen_start(self, final_mol: Chem.Mol) -> None:
        full_mol_conformer_id = rdDepictor.Compute2DCoords(final_mol)
        conformer = final_mol.GetConformer(full_mol_conformer_id)
        self.__final_mol = final_mol
        self.__mol_atom_idx_to_point2d = {}
        self.__unvisited_nodes = set(range(final_mol.GetNumAtoms()))
        for i in range(final_mol.GetNumAtoms()):
            self.__mol_atom_idx_to_point2d[i] = Point2D(conformer.GetAtomPosition(i))

        print(f"<h2>Bond Generation Steps</h2>", file=self.__out_fh)

    def render_partial_molecule_at_step(
        self,
        step: int,
        focus_node_idx: int,
        partial_molecule_adj_lists: List[np.ndarray],
        valid_edge_choices: List[Tuple[int, int]],
    ) -> str:
        step_mol = RWMol()
        for atom in self.__final_mol.GetAtoms():
            step_mol.AddAtom(atom)

        for type_idx, edges in enumerate(partial_molecule_adj_lists):
            for edge in edges[::2]:
                step_mol.AddBond(int(edge[0]), int(edge[1]), EDGE_TYPE_IDX_TO_BOND_TYPE[type_idx])

        # Now set labels for the unvisited / unconnected nodes:
        drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
        drawer_options = drawer.drawOptions()
        self.__unvisited_nodes.discard(focus_node_idx)
        for node_idx in self.__unvisited_nodes:
            drawer_options.atomLabels[
                node_idx
            ] = f"{node_idx}:{self.__final_mol.GetAtoms()[node_idx].GetSymbol()}"
        drawer.SetDrawOptions(drawer_options)

        # Draw the actual molecule, making sure to align it with the full molecule drawing:
        step_mol.UpdatePropertyCache()  # Triggers computation of properties required for drawing
        rdDepictor.Compute2DCoords(step_mol, coordMap=self.__mol_atom_idx_to_point2d)
        drawer.DrawMolecule(step_mol)

        # Draw a circle around the focus node:
        drawer.SetFillPolys(False)
        focus_node_2dpoint = self.__mol_atom_idx_to_point2d[focus_node_idx]
        drawer.DrawEllipse(
            Point2D(focus_node_2dpoint.x - 0.3, focus_node_2dpoint.y - 0.3),
            Point2D(focus_node_2dpoint.x + 0.3, focus_node_2dpoint.y + 0.3),
        )

        # Draw wavy lines indicating potential edges:
        for candidate_edge in valid_edge_choices:
            drawer.DrawWavyLine(
                self.__mol_atom_idx_to_point2d[candidate_edge[0]],
                self.__mol_atom_idx_to_point2d[candidate_edge[1]],
                color1=(0.7, 0, 0),  # Primary color is dark-ish red
                color2=(0, 0, 0),  # For the life of me, I don't know what color2 controls
                nSegments=3,
            )

        drawer.FinishDrawing()
        out_filename = f"partial_mol-step_{step}.png"
        drawer.WriteDrawingText(os.path.join(self.__out_dir, out_filename))
        return out_filename

    def render_molecule_gen_step(self, step: int, step_info: MoleculeGenerationStepInfo) -> None:
        print(
            f"<h3>Step {step} (focusing on node {step_info.focus_node_idx})</h3>",
            file=self.__out_fh,
        )

        # First, render the partial molecule:
        mol_rendering_path = self.render_partial_molecule_at_step(
            step,
            step_info.focus_node_idx,
            partial_molecule_adj_lists=step_info.partial_molecule_adjacency_lists,
            valid_edge_choices=[
                (step_info.focus_node_idx, candidate_edge.target_node_idx)
                for candidate_edge in step_info.candidate_edge_infos
            ],
        )
        print(f" <img src='{mol_rendering_path}' />", file=self.__out_fh)

        print(f"<table style='text-align:center'>", file=self.__out_fh)
        print(f" <tr>", file=self.__out_fh)
        print(f"  <th>Target Node</th>", file=self.__out_fh)
        print(f"  <th>Correctness</th>", file=self.__out_fh)
        print(f"  <th>Score</th>", file=self.__out_fh)
        print(f"  <th>Prob</th>", file=self.__out_fh)
        print(f"  <th>Single Bond Prob</th>", file=self.__out_fh)
        print(f"  <th>Double Bond Prob</th>", file=self.__out_fh)
        print(f"  <th>Triple Bond Prob</th>", file=self.__out_fh)
        print(f" </tr>", file=self.__out_fh)

        if len(step_info.candidate_edge_infos) > 0:
            max_logprob = max(edge_info.logprob for edge_info in step_info.candidate_edge_infos)
            if step_info.no_edge_logprob > max_logprob:
                max_logprob = step_info.no_edge_logprob
        else:
            max_logprob = step_info.no_edge_logprob

        # Now give information about the different choices:
        for edge_info in step_info.candidate_edge_infos:
            if edge_info.logprob >= max_logprob:
                print(f' <tr style="font-weight: bold">', file=self.__out_fh)
            else:
                print(f" <tr>", file=self.__out_fh)

            print(f"  <td>{edge_info.target_node_idx}</td>", file=self.__out_fh)
            if edge_info.correct is None:
                print(f"  <td>-</td>", file=self.__out_fh)
            else:
                print(f"  <td>{'C' if edge_info.correct else 'Inc'}orrect</td>", file=self.__out_fh)
            print(f"  <td>{edge_info.score:7.3f}</td>", file=self.__out_fh)
            print(f"  <td>{np.exp(edge_info.logprob):5.3f}</td>", file=self.__out_fh)
            print(f"  <td>{np.exp(edge_info.type_idx_to_logprobs[0]):.3f}</td>", file=self.__out_fh)
            print(f"  <td>{np.exp(edge_info.type_idx_to_logprobs[1]):.3f}</td>", file=self.__out_fh)
            print(f"  <td>{np.exp(edge_info.type_idx_to_logprobs[2]):.3f}</td>", file=self.__out_fh)
            print(f" </tr>", file=self.__out_fh)

        if step_info.no_edge_logprob >= max_logprob:
            print(f' <tr style="font-weight: bold">', file=self.__out_fh)
        else:
            print(f" <tr>", file=self.__out_fh)

        print(f"  <td>No Edge</td>", file=self.__out_fh)
        if step_info.no_edge_correct is None:
            print(f"  <td>-</td>", file=self.__out_fh)
        else:
            print(
                f"  <td>{'C' if step_info.no_edge_correct else 'Inc'}orrect</td>",
                file=self.__out_fh,
            )
        print(f"  <td>{step_info.no_edge_score:7.3f}</td>", file=self.__out_fh)
        print(f"  <td>{np.exp(step_info.no_edge_logprob):5.3f}</td>", file=self.__out_fh)
        print(f" </tr>", file=self.__out_fh)
        print(f"</table>", file=self.__out_fh)

    def visualise_from_smiles(self, smiles, out_dir: str):
        os.makedirs(out_dir)
        self.__out_dir = out_dir
        out_file = os.path.join(out_dir, "index.html")
        with open(out_file, "wt") as out_fh:
            self.__out_fh = out_fh
            print(f"<html><head><title>Generation of {smiles}</title></head>", file=self.__out_fh)
            print(f"<body>", file=self.__out_fh)
            print(f"<h1>Visualisation of CG-VAE generation of {smiles}</h1>", file=self.__out_fh)
            super().visualise_from_smiles(smiles)
            print(f"</body>", file=self.__out_fh)

        print(f"Written results to {out_file}")

    def visualise_from_samples(
        self, latents_path: str, node_representations: np.ndarray, out_dir: str
    ):
        os.makedirs(out_dir)
        self.__out_dir = out_dir
        out_file = os.path.join(out_dir, "index.html")
        with open(out_file, "wt") as out_fh:
            self.__out_fh = out_fh
            print(
                f"<html><head><title>Generation from latents in {latents_path}</title></head>",
                file=self.__out_fh,
            )
            print(f"<body>", file=self.__out_fh)
            print(
                f"<h1>Visualisation of CG-VAE generation from latents in {latents_path}</h1>",
                file=self.__out_fh,
            )
            super().visualise_from_samples(node_representations)
            print(f"</body>", file=self.__out_fh)

        print(f"Written results to {out_file}")


def run_from_args(args) -> None:
    visualiser = HTMLGraphGenerationVisualiser(args.MODEL_DIR)
    if os.path.exists(args.SMILES_OR_PATH):
        with open(args.SMILES_OR_PATH, "rb") as fh:
            samples = pickle.load(fh)
        visualiser.visualise_from_samples(args.SMILES_OR_PATH, samples, args.OUT_DIR)
    else:
        visualiser.visualise_from_smiles(args.SMILES_OR_PATH, args.OUT_DIR)


def run():
    parser = get_model_loading_parser(
        description="Visualise CGVAE molecule generation as HTML.", include_extra_args=False
    )
    parser.add_argument(
        "SMILES_OR_PATH",
        type=str,
        help="SMILES string or path of latent representations to visualise.",
    )
    parser.add_argument("OUT_DIR", type=str, help="Directory to store generated files.")
    parser.add_argument("--debug", dest="debug", action="store_true", help="Enable debug routines")
    args = parser.parse_args()

    # Shut up tensorflow:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    tf.get_logger().setLevel("ERROR")
    warnings.simplefilter("ignore")

    run_and_debug(lambda: run_from_args(args), args.debug)


if __name__ == "__main__":
    run()
