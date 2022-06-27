"""Decoding utilities for a MoLeR."""
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Tuple, NamedTuple, Iterable, DefaultDict, Union
import heapq

import numpy as np
import tensorflow as tf
from rdkit import Chem

from molecule_generation.chem.atom_feature_utils import AtomFeatureExtractor
from molecule_generation.chem.molecule_dataset_utils import BOND_DICT, featurise_atoms
from molecule_generation.chem.rdkit_helpers import initialise_atom_from_symbol, get_atom_symbol
from molecule_generation.chem.motif_utils import (
    MotifAnnotation,
    MotifAtomAnnotation,
    MotifVocabulary,
)
from molecule_generation.chem.topology_features import calculate_topology_features
from molecule_generation.chem.valence_constraints import (
    constrain_edge_choices_based_on_valence,
    constrain_edge_types_based_on_valence,
)
from molecule_generation.preprocessing.cgvae_generation_trace import (
    calculate_dist_from_focus_to_valid_target,
)
from molecule_generation.preprocessing.moler_generation_trace import (
    Edge,
    get_open_attachment_points,
)


EDGE_TYPE_IDX_TO_BOND_TYPE = {
    0: Chem.BondType.SINGLE,
    1: Chem.BondType.DOUBLE,
    2: Chem.BondType.TRIPLE,
}


class DecoderSamplingMode(Enum):
    GREEDY = 0
    SAMPLING = 1


def sample_indices_from_logprobs(
    num_samples: int, sampling_mode: DecoderSamplingMode, logprobs: np.ndarray
) -> Iterable[int]:
    """Samples indices (without replacement) given the log-likelihoods.

    Args:
        num_samples: intended number of samples
        sampling_mode: sampling method (greedy
        logprobs: log-probabilities of selecting appropriate entries

    Returns:
        indices of picked values, shape (n,), where n = min(num_samples, available_samples)

    Note:
        the ordering of returned indices is arbitrary
    """
    num_choices = logprobs.shape[0]
    indices = np.arange(num_choices)
    num_samples = min(num_samples, num_choices)  # Handle cases where we only have few candidates
    if sampling_mode == DecoderSamplingMode.GREEDY:
        # Note that this will return the top num_samples indices, but not in order:
        picked_indices = np.argpartition(logprobs, -num_samples)[-num_samples:]
    elif sampling_mode == DecoderSamplingMode.SAMPLING:
        p = np.exp(logprobs)  # Convert to probabilities
        # We can only sample values with non-zero probabilities
        num_choices = np.sum(p > 0)
        num_samples = min(num_samples, num_choices)
        picked_indices = np.random.choice(
            indices,
            size=(num_samples,),
            replace=False,
            p=p,
        )
    else:
        raise ValueError(f"Sampling method {sampling_mode} not known.")

    return picked_indices


class MoleculeGenerationEdgeCandidateInfo(NamedTuple):
    target_node_idx: int
    score: float
    logprob: float
    correct: Optional[bool]
    type_idx_to_logprobs: np.ndarray


@dataclass
class MoleculeGenerationAtomChoiceInfo:
    node_idx: int
    true_type_idx: Optional[List[int]]
    type_idx_to_prob: List[float]
    molecule: Optional[Chem.Mol] = None


@dataclass
class MoleculeGenerationEdgeChoiceInfo:
    focus_node_idx: int
    partial_molecule_adjacency_lists: List[np.ndarray]
    candidate_edge_infos: List[MoleculeGenerationEdgeCandidateInfo]
    no_edge_score: float
    no_edge_logprob: float
    no_edge_correct: Optional[bool]
    molecule: Optional[Chem.Mol] = None


@dataclass
class MoleculeGenerationAttachmentPointChoiceInfo:
    partial_molecule_adjacency_lists: List[np.ndarray]
    motif_nodes: List[int]
    candidate_attachment_points: List[int]
    candidate_idx_to_prob: List[float]
    correct_attachment_point_idx: Optional[int]
    molecule: Optional[Chem.Mol] = None


MoleculeGenerationChoiceInfo = Union[
    MoleculeGenerationAtomChoiceInfo,
    MoleculeGenerationEdgeChoiceInfo,
    MoleculeGenerationAttachmentPointChoiceInfo,
]


class MoLeRDecoderState(object):
    """
    Data class holding the state of the MoLeR decoder.

    Note: To enable safe usage in contexts with sharing (such as beam search), the
    contract is that mutable members of this class cannot be manipulated directly,
    but need to be copied on modification. Hence, a number of static creation
    methods (new_with_*) are provided which should enable all necessary actions.
    """

    def __init__(
        self,
        molecule_representation: tf.Tensor,
        molecule_id: Any,
        logprob: float = 0.0,
        molecule: Optional[Chem.Mol] = None,
        atom_types: Optional[List[str]] = None,
        adjacency_lists: Optional[List[List[Tuple[int, int]]]] = None,
        visited_atoms: Optional[List[int]] = None,
        atoms_to_visit: Optional[List[int]] = None,
        atoms_to_mark_as_visited: Optional[List[int]] = None,
        focus_atom: Optional[int] = None,
        prior_focus_atom: Optional[int] = None,
        generation_steps: Optional[List[MoleculeGenerationChoiceInfo]] = None,
        candidate_attachment_points: Optional[List[int]] = None,
        motifs: Optional[List[MotifAnnotation]] = None,
        num_free_bond_slots: Optional[List[Optional[int]]] = None,
    ):
        """
        Create new state tracking generation of a molecule.

        Args:
            molecule_representation: Representation from which this molecule
                is being generated.
            molecule_id: Object used to identify this particular input; usually
                set at the beginning of the decoding process and then simply
                propagated.
            logprob: Log probability of the current decoder state, i.e., the sum
                of the log probabilities of all steps leading to it from the
                initial start.
            molecule: An optional (partial) molecule encoding the current
                state of the generation. If it is missing, we start from an
                empty molecule, and all remaining arguments have to be None
                as well.
            atom_types: A list of the atom types (string representation including
                charges, e.g., "N++") used in the passed partial molecule. This has
                to be passed in if the given partial molecule is non-empty.
            adjacency_lists: _Symmetric_ adjacency lists (i.e., we assume
                that both (u, v) and (v, u) are provided), corresponding to the
                partial molecule we are decoding, as a list of int32 tensors of
                shape [E, 2].
            visited_atoms: Atoms in the molecule (stored as indices) that we have
                already processed and to which we can add fresh connections.
            atoms_to_visit: Atoms in the molecule (stored as indices) that we still
                need to process and from which we can add new bonds.
            atoms_to_mark_as_visited: Atoms in the molecule (stored as indices) that we want to mark
                as visited when `new_with_focus_marked_as_visited` is called. This is used to delay
                marking as visited the motif nodes, in order to avoid adding additional edges within
                motifs.
            focus_atom: Atom that we are currently focusing on, i.e., from which
                we generate new bonds.
            prior_focus_atom: The focus atom that we finished last. This is required
                to inform the next-atom choice...
            generation_steps: List of generation step metadata, or `None` if we don't want to store
                the generation traces.
            candidate_attachment_points: Atoms in the molecule (stored as indices) that are
                candidates to consider for picking an attachment point.
            motifs: List of annotations of motifs that already exist in the current molecule.
            num_free_bond_slots: Number of bonds that can still be attached to each existing atom.
                Passing `None` for a specific atom means there are no constraints for it, passing
                `None` for `num_free_bond_slots` itself means there are no constraints at all.

        """
        self._molecule_representation = molecule_representation
        self._molecule_id = molecule_id
        self._logprob = logprob

        if molecule is None and not (
            atom_types is None
            and adjacency_lists is None
            and visited_atoms is None
            and atoms_to_visit is None
            and focus_atom is None
            and prior_focus_atom is None
        ):
            raise ValueError(
                "Molecule generation states created without a partial molecule cannot take any other values."
            )

        self._molecule = molecule or Chem.RWMol()
        if self._molecule.GetNumAtoms() > 0 and atom_types is None:
            raise ValueError(
                f"Molecule generation states starting with a partial molecule require atom type information!"
            )
        self._atom_types: List[str] = atom_types or []
        self._adjacency_lists = adjacency_lists or [[] for _ in range(len(BOND_DICT))]
        self._visited_atoms: List[int] = visited_atoms or []
        self._atoms_to_visit: List[int] = atoms_to_visit or []
        self._atoms_to_mark_as_visited: List[int] = atoms_to_mark_as_visited or []
        self._focus_atom: Optional[int] = focus_atom
        self._prior_focus_atom: Optional[int] = prior_focus_atom
        self._generation_steps = generation_steps
        self._candidate_attachment_points = candidate_attachment_points or []
        self._motifs = motifs or []
        self._num_free_bond_slots = num_free_bond_slots or [None] * len(atom_types)

    @staticmethod
    def extend_generation_steps(
        generation_steps: Optional[List[MoleculeGenerationChoiceInfo]],
        choice_info: MoleculeGenerationChoiceInfo,
        molecule: Chem.Mol,
    ):
        if choice_info is None:
            return generation_steps
        else:
            assert generation_steps is not None
            choice_info.molecule = Chem.Mol(molecule)

            return list(generation_steps) + [choice_info]

    @staticmethod
    def new_with_added_atom(
        old_state: "MoLeRDecoderState",
        atom_symbol: str,
        atom_logprob: float,
        atom_choice_info: Optional[MoleculeGenerationAtomChoiceInfo] = None,
    ) -> "MoLeRDecoderState":
        """Add a new atom to the partial mocule under construction.

        Note: This also sets the focus node to the next atom to consider, chosen from the
        new ones added.
        """
        if old_state._focus_atom is not None:
            raise ValueError(
                "New atoms can only be added when the decoder is not focused on creating bonds for an atom!"
            )
        new_mol = Chem.RWMol(old_state._molecule)
        new_atom_idx = new_mol.AddAtom(initialise_atom_from_symbol(atom_symbol))
        new_atom_types = list(old_state._atom_types)
        new_atom_types.append(atom_symbol)
        new_atoms_to_visit = list(old_state._atoms_to_visit)
        new_atoms_to_visit.append(new_atom_idx)
        new_focus_atom = new_atoms_to_visit.pop(0)  # BFS exploration; .pop() would give DFS

        new_generation_steps = MoLeRDecoderState.extend_generation_steps(
            generation_steps=old_state._generation_steps,
            choice_info=atom_choice_info,
            molecule=new_mol,
        )

        return MoLeRDecoderState(
            molecule_representation=old_state._molecule_representation,
            molecule_id=old_state._molecule_id,
            logprob=old_state._logprob + atom_logprob,
            molecule=new_mol,
            atom_types=new_atom_types,
            adjacency_lists=old_state._adjacency_lists,
            visited_atoms=old_state._visited_atoms,
            atoms_to_visit=new_atoms_to_visit,
            atoms_to_mark_as_visited=old_state._atoms_to_mark_as_visited,
            focus_atom=new_focus_atom,
            prior_focus_atom=old_state._focus_atom,
            generation_steps=new_generation_steps,
            candidate_attachment_points=old_state._candidate_attachment_points,
            motifs=old_state._motifs,
            num_free_bond_slots=old_state._num_free_bond_slots + [None],
        )

    @staticmethod
    def new_for_finished_decoding(
        old_state: "MoLeRDecoderState",
        finish_logprob: float,
        atom_choice_info: Optional[MoleculeGenerationAtomChoiceInfo] = None,
    ) -> "MoLeRDecoderState":
        """Create a new state for a finished decoding run."""
        if old_state._focus_atom is not None:
            raise ValueError(
                "Decoding can only be finished when the decoder is not focused on creating bonds for an atom!"
            )

        new_generation_steps = MoLeRDecoderState.extend_generation_steps(
            generation_steps=old_state._generation_steps,
            choice_info=atom_choice_info,
            molecule=old_state._molecule,
        )

        return MoLeRDecoderState(
            molecule_representation=old_state._molecule_representation,
            molecule_id=old_state._molecule_id,
            logprob=old_state._logprob + finish_logprob,
            molecule=old_state._molecule,
            atom_types=old_state._atom_types,
            adjacency_lists=old_state._adjacency_lists,
            visited_atoms=old_state._visited_atoms,
            atoms_to_visit=old_state._atoms_to_visit,
            atoms_to_mark_as_visited=old_state._atoms_to_mark_as_visited,
            focus_atom=-1,
            prior_focus_atom=-1,
            generation_steps=new_generation_steps,
            candidate_attachment_points=old_state._candidate_attachment_points,
            motifs=old_state._motifs,
            num_free_bond_slots=old_state._num_free_bond_slots,
        )

    @staticmethod
    def new_with_added_bond(
        old_state: "MoLeRDecoderState",
        target_atom_idx: int,
        bond_type_idx: int,
        bond_logprob: float,
        edge_choice_info: Optional[MoleculeGenerationEdgeChoiceInfo] = None,
    ) -> "MoLeRDecoderState":
        """Add a new bond to the partial mocule under construction."""
        if old_state._focus_atom is None:
            raise ValueError(
                "New bonds can only be added when the decoder is focused on creating bonds for an atom!"
            )
        new_mol = Chem.RWMol(old_state._molecule)
        new_mol.AddBond(
            old_state._focus_atom, target_atom_idx, EDGE_TYPE_IDX_TO_BOND_TYPE[bond_type_idx]
        )
        # We only need to make an actual copy of the adj_list for the bond type we change:
        new_adjacency_lists = list(old_state._adjacency_lists)
        new_adjacency_lists[bond_type_idx] = list(new_adjacency_lists[bond_type_idx])
        new_adjacency_lists[bond_type_idx].append((old_state._focus_atom, target_atom_idx))
        new_adjacency_lists[bond_type_idx].append((target_atom_idx, old_state._focus_atom))

        new_generation_steps = MoLeRDecoderState.extend_generation_steps(
            generation_steps=old_state._generation_steps,
            choice_info=edge_choice_info,
            molecule=new_mol,
        )

        if old_state._num_free_bond_slots[old_state._focus_atom] is not None:
            raise ValueError("Focus atom has a constraint on the number of new bonds.")

        if old_state._num_free_bond_slots[target_atom_idx] == 0:
            raise ValueError("Tried to attach to an atom with no free bond slots.")

        new_num_free_bond_slots = list(old_state._num_free_bond_slots)

        # If we're connecting to an atom that has a constraint on the number of bonds, decrease that
        # by 1. Note that we don't differentiate single/double/triple bonds here.
        if new_num_free_bond_slots[target_atom_idx] is not None:
            new_num_free_bond_slots[target_atom_idx] -= 1

        return MoLeRDecoderState(
            molecule_representation=old_state._molecule_representation,
            molecule_id=old_state._molecule_id,
            logprob=old_state._logprob + bond_logprob,
            molecule=new_mol,
            atom_types=old_state._atom_types,
            adjacency_lists=new_adjacency_lists,
            visited_atoms=old_state._visited_atoms,
            atoms_to_visit=old_state._atoms_to_visit,
            atoms_to_mark_as_visited=old_state._atoms_to_mark_as_visited,
            focus_atom=old_state._focus_atom,
            prior_focus_atom=old_state._prior_focus_atom,
            generation_steps=new_generation_steps,
            candidate_attachment_points=old_state._candidate_attachment_points,
            motifs=old_state._motifs,
            num_free_bond_slots=new_num_free_bond_slots,
        )

    @staticmethod
    def new_with_added_motif(
        old_state: "MoLeRDecoderState",
        motif_type: str,
        motif_logprob: float,
        atom_choice_info: Optional[MoleculeGenerationAtomChoiceInfo] = None,
    ) -> "MoLeRDecoderState":
        """Add a new atom or entire motif to the partial mocule under construction."""
        motif = Chem.MolFromSmiles(motif_type)
        Chem.rdmolops.Kekulize(motif, clearAromaticFlags=True)

        motif_atom_symbols = [get_atom_symbol(atom) for atom in motif.GetAtoms()]
        motif_num_atoms = len(motif_atom_symbols)

        node_idx_offset = len(old_state.molecule.GetAtoms())

        motif_adjacency_list = [
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), BOND_DICT[str(bond.GetBondType())])
            for bond in motif.GetBonds()
        ]

        # Bonds to add to the motif, grouped by the larger of endpoint ids.
        motif_bonds = [[] for _ in range(motif_num_atoms)]

        for node_start, node_end, bond_type in motif_adjacency_list:
            if node_start > node_end:
                node_start, node_end = node_end, node_start

            motif_bonds[node_end].append((node_start, bond_type))

        assert not old_state._atoms_to_visit
        assert old_state._focus_atom is None

        current_state = old_state

        for atom_symbol, bonds in zip(motif_atom_symbols, motif_bonds):
            current_state._focus_atom = None

            current_state = MoLeRDecoderState.new_with_added_atom(
                old_state=current_state,
                atom_symbol=atom_symbol,
                atom_logprob=0.0,  # We will add the full motif logprob at the end.
            )

            for target_atom_idx, bond_type in bonds:
                current_state = MoLeRDecoderState.new_with_added_bond(
                    old_state=current_state,
                    target_atom_idx=node_idx_offset + target_atom_idx,
                    bond_type_idx=bond_type,
                    bond_logprob=0.0,  # We will add the full motif logprob at the end
                )

        motif_nodes = list(range(node_idx_offset, node_idx_offset + motif_num_atoms))

        motif_node_symmetry_class = Chem.CanonicalRankAtoms(motif, breakTies=False)

        attachment_points = []
        seen_symmetry_classes = set()

        for node_idx, symmetry_class in enumerate(motif_node_symmetry_class):
            if symmetry_class not in seen_symmetry_classes:
                seen_symmetry_classes.add(symmetry_class)
                attachment_points.append(node_idx)

        attachment_points = get_open_attachment_points(
            valid_attachment_points=attachment_points,
            adjacency_list=[Edge(*args) for args in motif_adjacency_list],
            node_types=motif_atom_symbols,
        )

        attachment_points = [node_idx + node_idx_offset for node_idx in attachment_points]

        new_generation_steps = MoLeRDecoderState.extend_generation_steps(
            generation_steps=old_state._generation_steps,
            choice_info=atom_choice_info,
            molecule=current_state._molecule,
        )

        # Record the freshly added motif, so it can be marked in the partial graph node features.
        # Note that `symmetry_class_id` is not used here.
        new_motifs = list(old_state._motifs)
        new_motifs.append(
            MotifAnnotation(
                motif_type=motif_type,
                atoms=[
                    MotifAtomAnnotation(atom_id=atom_id, symmetry_class_id=-1)
                    for atom_id in motif_nodes
                ],
            )
        )

        return MoLeRDecoderState(
            molecule_representation=current_state._molecule_representation,
            molecule_id=old_state._molecule_id,
            logprob=old_state._logprob + motif_logprob,
            molecule=current_state._molecule,
            atom_types=current_state._atom_types,
            adjacency_lists=current_state._adjacency_lists,
            visited_atoms=current_state._visited_atoms,
            atoms_to_visit=None,
            atoms_to_mark_as_visited=motif_nodes,
            focus_atom=None,
            prior_focus_atom=old_state._prior_focus_atom,
            generation_steps=new_generation_steps,
            candidate_attachment_points=attachment_points,
            motifs=new_motifs,
            num_free_bond_slots=old_state._num_free_bond_slots + [None] * motif_num_atoms,
        )

    @staticmethod
    def new_with_focus_on_attachment_point(
        old_state: "MoLeRDecoderState",
        new_focus_atom: int,
        focus_atom_logprob: float,
        attachment_point_choice_info: MoleculeGenerationAttachmentPointChoiceInfo,
    ) -> "MoLeRDecoderState":
        new_generation_steps = MoLeRDecoderState.extend_generation_steps(
            generation_steps=old_state._generation_steps,
            choice_info=attachment_point_choice_info,
            molecule=old_state._molecule,
        )

        return MoLeRDecoderState(
            molecule_representation=old_state._molecule_representation,
            molecule_id=old_state._molecule_id,
            logprob=old_state._logprob + focus_atom_logprob,
            molecule=old_state._molecule,
            atom_types=old_state._atom_types,
            adjacency_lists=old_state._adjacency_lists,
            visited_atoms=old_state._visited_atoms,
            atoms_to_visit=old_state._atoms_to_visit,
            atoms_to_mark_as_visited=old_state._atoms_to_mark_as_visited,
            focus_atom=new_focus_atom,
            prior_focus_atom=old_state._focus_atom,
            generation_steps=new_generation_steps,
            candidate_attachment_points=old_state._candidate_attachment_points,
            motifs=old_state._motifs,
            num_free_bond_slots=old_state._num_free_bond_slots,
        )

    @staticmethod
    def new_with_focus_marked_as_visited(
        old_state: "MoLeRDecoderState",
        focus_node_finished_logprob: float,
        edge_choice_info: Optional[MoleculeGenerationEdgeChoiceInfo] = None,
    ) -> "MoLeRDecoderState":
        """Mark a focus node as visited. If more nodes are available to visit, directly resets
        the focus node to one of them.
        """
        if old_state._focus_atom is None:
            raise ValueError(
                "Focus can only be marked as visited when the decoder is focused on creating bonds for an atom!"
            )
        new_visited_atoms = list(old_state._visited_atoms)
        new_visited_atoms.append(old_state._focus_atom)

        if len(old_state._atoms_to_visit) > 0:
            new_atoms_to_visit = list(old_state._atoms_to_visit)
            new_focus_atom: Optional[int] = new_atoms_to_visit.pop(
                0
            )  # BFS exploration; .pop() would give DFS
        else:
            new_atoms_to_visit = old_state._atoms_to_visit
            new_focus_atom = None

        if old_state._atoms_to_mark_as_visited:
            # The last node addition step added a motif - we now need to mark all motif nodes as visited.

            count_already_in_visited = 0
            for node_idx in old_state._atoms_to_mark_as_visited:
                if node_idx in new_visited_atoms:
                    count_already_in_visited += 1
                else:
                    new_visited_atoms.append(node_idx)

            # The should be one motif node already added as visited: the attachment point.
            assert count_already_in_visited == 1

        new_generation_steps = MoLeRDecoderState.extend_generation_steps(
            generation_steps=old_state._generation_steps,
            choice_info=edge_choice_info,
            molecule=old_state._molecule,
        )

        return MoLeRDecoderState(
            molecule_representation=old_state._molecule_representation,
            molecule_id=old_state._molecule_id,
            logprob=old_state._logprob + focus_node_finished_logprob,
            molecule=old_state._molecule,
            atom_types=old_state._atom_types,
            adjacency_lists=old_state._adjacency_lists,
            visited_atoms=new_visited_atoms,
            atoms_to_visit=new_atoms_to_visit,
            atoms_to_mark_as_visited=None,
            focus_atom=new_focus_atom,
            prior_focus_atom=old_state._focus_atom,
            generation_steps=new_generation_steps,
            candidate_attachment_points=old_state._candidate_attachment_points,
            motifs=old_state._motifs,
            num_free_bond_slots=old_state._num_free_bond_slots,
        )

    @property
    def molecule(self):
        return self._molecule

    @property
    def molecule_representation(self):
        return self._molecule_representation

    @property
    def molecule_id(self):
        return self._molecule_id

    @property
    def logprob(self) -> float:
        return self._logprob

    @property
    def focus_atom(self):
        return self._focus_atom

    @property
    def prior_focus_atom(self):
        return self._prior_focus_atom

    @property
    def adjacency_lists(self):
        return self._adjacency_lists

    @property
    def atoms_to_visit(self):
        return self._atoms_to_visit

    @property
    def atoms_to_mark_as_visited(self):
        return self._atoms_to_mark_as_visited

    @property
    def generation_steps(self):
        return self._generation_steps

    @property
    def candidate_attachment_points(self):
        return self._candidate_attachment_points

    def get_node_features(
        self,
        atom_featurisers: List[AtomFeatureExtractor],
        motif_vocabulary: Optional[MotifVocabulary] = None,
    ):
        """
        Compute node features for consumption in a GNN from the partial molecule
        in this decoder state.
        """
        # The overall computation here should not be cached, as we change the molecule
        # between calls (adding atoms/bonds) and hence would never expect the same
        # return values.
        self._molecule.UpdatePropertyCache(strict=False)

        features = featurise_atoms(
            mol=self._molecule,
            atom_feature_extractors=atom_featurisers,
            motif_vocabulary=motif_vocabulary,
            motifs=self._motifs,
        )

        return np.concatenate([features.real_valued_features]), features.categorical_features

    def get_bond_candidate_targets(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute candidates for bonds from the current focus atom.

        Returns:
            Pair of two numpy arrays.
            First is a list of all valid targets of bonds from the current focus node. Shape: [CE]
            Second is a mask of allowed bond types for the valid targets. Shape: [CE, 3]
        """

        # We can only make bonds to nodes we have visited, but not exhausted the maximum number of
        # bonds; this is our initial target set:
        candidate_targets = {
            idx for idx in self._visited_atoms if self._num_free_bond_slots[idx] != 0
        }

        # However, we remove those that we have already connected to:
        for adj_list in self._adjacency_lists:
            for edge in adj_list:
                if edge[0] == self._focus_atom:
                    candidate_targets.discard(edge[1])
        candidate_targets = np.array(list(candidate_targets), dtype=np.int32)

        # Go from list of lists of tuples to list of numpy arrays, as we use slicing later:
        adjacency_lists = [np.array(adj_list) for adj_list in self._adjacency_lists]

        # Figure out which of these connections are chemically invalid:
        candidate_targets_mask = constrain_edge_choices_based_on_valence(
            start_node=self._focus_atom,
            candidate_target_nodes=candidate_targets,
            adjacency_lists=adjacency_lists,
            node_types=self._atom_types,
        )

        bond_type_mask = constrain_edge_types_based_on_valence(
            start_node=self._focus_atom,
            candidate_target_nodes=candidate_targets,
            # Go from list of lists of tuples to list of numpy arrays, as we use slicing inside:
            adjacency_lists=adjacency_lists,
            node_types=self._atom_types,
        )

        return candidate_targets[candidate_targets_mask], bond_type_mask[candidate_targets_mask]

    def compute_bond_candidate_features(self, candidate_targets) -> np.ndarray:
        """
        Compute features for bond candidates.

        Note: This has to match

        Args:
            candidate_targets: NumPy array of bond targets from the current focus node.
                Shape [EC], dtype int32

        Returns:
            NumPy array of features, where the i-th row of features corresponds to the
                i-th bond candidate. Shape [EC, EFD], dtype int32
        """
        # Compute distance features:
        distance_features: List[int] = calculate_dist_from_focus_to_valid_target(
            adjacency_list=self._adjacency_lists,
            focus_node=self._focus_atom,
            target_nodes=candidate_targets,
            symmetrise_adjacency_list=False,
        )  # Shape [EC]
        # Calculate the topology featues:
        topology_features = calculate_topology_features(
            edges=[(self._focus_atom, target) for target in candidate_targets], mol=self._molecule
        )  # Shape [EC, 2]

        return np.concatenate(
            [np.expand_dims(distance_features, axis=-1), topology_features], axis=1
        )


def restrict_to_beam_size_per_mol(
    decoder_states: Iterable[MoLeRDecoderState], beam_size: int
) -> List[MoLeRDecoderState]:
    grouped_by_mol: DefaultDict[Any, List[MoLeRDecoderState]] = defaultdict(list)
    for decoder_state in decoder_states:
        grouped_by_mol[decoder_state.molecule_id].append(decoder_state)
    restricted_decoder_states = []
    for per_mol_decoder_states in grouped_by_mol.values():
        restricted_decoder_states.extend(
            heapq.nlargest(beam_size, per_mol_decoder_states, key=lambda s: s.logprob)
        )
    return restricted_decoder_states
