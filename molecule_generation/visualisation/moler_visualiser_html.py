import argparse
import os
import pickle
from typing import List, Dict, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import BondType, RWMol, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry.rdGeometry import Point2D

from molecule_generation.utils.moler_visualisation_utils import (
    GraphGenerationVisualiser,
    PropertyPredictionInformation,
    MoleculeGenerationAtomChoiceInfo,
    MoleculeGenerationEdgeChoiceInfo,
    MoleculeGenerationAttachmentPointChoiceInfo,
)
from molecule_generation.utils.cli_utils import get_model_loading_parser


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

    def render_atom_data(
        self,
        atom_info: MoleculeGenerationAtomChoiceInfo,
        choice_descr: str,
        prob_threshold: float = 0.001,
    ) -> None:
        print(f"<h3>New Atom/Motif Choice</h3>", file=self.__out_fh)
        print(
            f"Selecting atom/motif for {choice_descr}, based on molecule generated so far.",
            file=self.__out_fh,
        )
        if prob_threshold > 0:
            print(
                f"Only showing choices with assigned probability >= {prob_threshold}.",
                file=self.__out_fh,
            )

        if atom_info.true_type_idx is not None:
            print(f"</br>Correct choices: ", file=self.__out_fh)
            print(
                f", ".join(
                    [
                        self.dataset._node_type_index_to_string[idx]
                        for idx in atom_info.true_type_idx
                    ]
                ),
                file=self.__out_fh,
            )
            print(f"</br>", file=self.__out_fh)

        print(f"</br>", file=self.__out_fh)

        # Show all possible choices, limited to those over threshold:
        print(f"<table>", file=self.__out_fh)
        print(f" <tr>", file=self.__out_fh)
        print(f"  <th>Atom/Motif</th>", file=self.__out_fh)
        print(f"  <th>Predicted Probability</th>", file=self.__out_fh)
        print(f" </tr>", file=self.__out_fh)

        max_prob_atom_idx = np.argmax(atom_info.type_idx_to_prob)
        for idx, prob, descr in self.get_atom_and_motif_types_to_render(atom_info, prob_threshold):
            print(f" <tr>", file=self.__out_fh)
            if idx == max_prob_atom_idx:
                print(f"  <td><b>{descr}</b></td>", file=self.__out_fh)
                print(f"  <td><b>{prob:.3f}</b></td>", file=self.__out_fh)
            else:
                print(f"  <td>{descr}</td>", file=self.__out_fh)
                print(f"  <td>{prob:.3f}</td>", file=self.__out_fh)
            print(f" </tr>", file=self.__out_fh)
        print(f"</table>", file=self.__out_fh)

    def render_attachment_point_selection_step(
        self, step: int, attachment_point_info: MoleculeGenerationAttachmentPointChoiceInfo
    ) -> None:
        print("<hr>", file=self.__out_fh)
        print(f"<h2>Step {step}</h2>", file=self.__out_fh)

        # First, render the partial molecule:
        mol_rendering_path = self.render_partial_molecule_at_step(
            step,
            new_visited_nodes_ids=attachment_point_info.motif_nodes,
            highlight_nodes_ids=attachment_point_info.candidate_attachment_points,
            partial_molecule_adj_lists=attachment_point_info.partial_molecule_adjacency_lists,
            valid_edge_choices=[],
        )
        print(f" <img src='{mol_rendering_path}' />", file=self.__out_fh)

        print(f"<table style='text-align:center'>", file=self.__out_fh)
        print(f" <tr>", file=self.__out_fh)
        print(f"  <th>Attachment Point</th>", file=self.__out_fh)
        print(f"  <th>Correctness</th>", file=self.__out_fh)
        print(f"  <th>Prob</th>", file=self.__out_fh)
        print(f" </tr>", file=self.__out_fh)

        max_prob = max(attachment_point_info.candidate_idx_to_prob)

        # Now give information about the different attachment point choices:
        for node_idx, prob in zip(
            attachment_point_info.candidate_attachment_points,
            attachment_point_info.candidate_idx_to_prob,
        ):
            if prob >= max_prob:
                print(f' <tr style="font-weight: bold">', file=self.__out_fh)
            else:
                print(f" <tr>", file=self.__out_fh)

            print(f"  <td>{node_idx}</td>", file=self.__out_fh)

            if attachment_point_info.correct_attachment_point_idx is None:
                print(f"  <td>-</td>", file=self.__out_fh)
            else:
                correct = node_idx == attachment_point_info.correct_attachment_point_idx
                print(f"  <td>{'C' if correct else 'Inc'}orrect</td>", file=self.__out_fh)

            print(f"  <td>{prob:5.3f}</td>", file=self.__out_fh)
            print(f" </tr>", file=self.__out_fh)

        print(f"</table>", file=self.__out_fh)

    def render_molecule_gen_start(self, final_mol: Chem.Mol) -> None:
        full_mol_conformer_id = rdDepictor.Compute2DCoords(final_mol)
        conformer = final_mol.GetConformer(full_mol_conformer_id)
        self.__final_mol = final_mol
        self.__mol_atom_idx_to_point2d = {}
        self.__unvisited_nodes = set(range(final_mol.GetNumAtoms()))
        self.__visited_nodes = set()
        for i in range(final_mol.GetNumAtoms()):
            self.__mol_atom_idx_to_point2d[i] = Point2D(conformer.GetAtomPosition(i))

        print(f"<h2>Molecule Generation Steps</h2>", file=self.__out_fh)

    def render_partial_molecule_at_step(
        self,
        step: int,
        new_visited_nodes_ids: List[int],
        highlight_nodes_ids: List[int],
        partial_molecule_adj_lists: List[np.ndarray],
        valid_edge_choices: List[Tuple[int, int]],
    ) -> str:
        step_mol = RWMol()
        for atom in self.__final_mol.GetAtoms():
            step_mol.AddAtom(atom)

        for type_idx, edges in enumerate(partial_molecule_adj_lists):
            for source, target in edges:
                if source < target:
                    step_mol.AddBond(int(source), int(target), EDGE_TYPE_IDX_TO_BOND_TYPE[type_idx])

        # Now set labels for the visited nodes:
        drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
        drawer_options = drawer.drawOptions()
        self.__visited_nodes.update(map(int, new_visited_nodes_ids))  # np.int32 to pyInt
        for node_idx in self.__visited_nodes:
            drawer_options.atomLabels[
                node_idx
            ] = f"{node_idx}:{self.__final_mol.GetAtoms()[node_idx].GetSymbol()}"

        # And slightly cleaner ones for the unvisited nodes:
        self.__unvisited_nodes.difference_update(
            new_visited_nodes_ids
        )  # Python auto converts np.int32 -> pyInt
        for node_idx in self.__unvisited_nodes:
            drawer_options.atomLabels[node_idx] = self.__final_mol.GetAtoms()[node_idx].GetSymbol()

        drawer.SetDrawOptions(drawer_options)

        # Draw the actual molecule, making sure to align it with the full molecule drawing:
        step_mol.UpdatePropertyCache()  # Triggers computation of properties required for drawing
        rdDepictor.Compute2DCoords(step_mol, coordMap=self.__mol_atom_idx_to_point2d)
        drawer.DrawMolecule(step_mol)

        # Draw circles around highlighted nodes:
        drawer.SetFillPolys(False)

        for highlight_node_idx in highlight_nodes_ids:
            node_2dpoint = self.__mol_atom_idx_to_point2d[highlight_node_idx]
            drawer.DrawEllipse(
                Point2D(node_2dpoint.x - 0.3, node_2dpoint.y - 0.3),
                Point2D(node_2dpoint.x + 0.3, node_2dpoint.y + 0.3),
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

    def render_molecule_gen_edge_step(
        self, step: int, step_info: MoleculeGenerationEdgeChoiceInfo
    ) -> None:
        print("<hr>", file=self.__out_fh)
        print(
            f"<h2>Step {step}</h2><h3>(focusing on node {step_info.focus_node_idx})</h3>",
            file=self.__out_fh,
        )

        # First, render the partial molecule:
        mol_rendering_path = self.render_partial_molecule_at_step(
            step,
            new_visited_nodes_ids=[step_info.focus_node_idx],
            highlight_nodes_ids=[step_info.focus_node_idx],
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
            print(f"<h1>Visualisation of MoLeR generation of {smiles}</h1>", file=self.__out_fh)
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
                f"<h1>Visualisation of MoLeR generation from latents in {latents_path}</h1>",
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


def get_argparser() -> argparse.ArgumentParser():
    parser = get_model_loading_parser(
        description="Visualise MoLeR molecule generation as HTML.", include_extra_args=False
    )
    parser.add_argument(
        "SMILES_OR_PATH",
        type=str,
        help="SMILES string or path of latent representations to visualise.",
    )
    parser.add_argument("OUT_DIR", type=str, help="Directory to store generated files.")
    return parser
