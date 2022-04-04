import logging
from collections import Counter
from typing import Dict, List, NamedTuple, Tuple, Optional

from rdkit import Chem


logger = logging.getLogger(__name__)


class MotifExtractionSettings(NamedTuple):
    min_frequency: Optional[int]
    min_num_atoms: int
    cut_leaf_edges: bool
    max_vocab_size: Optional[int] = None


class MotifAtomAnnotation(NamedTuple):
    """Atom id together with the id of its symmetry classes inside the motif."""

    atom_id: int
    symmetry_class_id: int


class MotifAnnotation(NamedTuple):
    """Information about a specific occurrence of a motif in a molecule."""

    motif_type: str
    atoms: List[MotifAtomAnnotation]


class MotifVocabulary(NamedTuple):
    """Vocabulary of motifs, together with settings used to extract it."""

    vocabulary: Dict[str, int]
    settings: MotifExtractionSettings


def get_bonds_to_fragment_on(molecule: Chem.Mol, cut_leaf_edges: bool) -> List[int]:
    """Get a list of bond ids to cut for extracting candidate motifs.

    This returns all bonds (u, v) such that:
        - (u, v) does not lie on a ring
        - either u or v lies on a ring

    Additionally, if `cut_leaf_edges` is `False`, edges leading to nodes of degree 1 are skipped.

    Returns:
        List of ids of bonds that should be cut.
    """
    ids_of_bonds_to_cut = []

    for bond in molecule.GetBonds():
        if bond.IsInRing():
            continue

        atom_begin = bond.GetBeginAtom()
        atom_end = bond.GetEndAtom()

        if not cut_leaf_edges and min(atom_begin.GetDegree(), atom_end.GetDegree()) == 1:
            continue

        if not atom_begin.IsInRing() and not atom_end.IsInRing():
            continue

        ids_of_bonds_to_cut.append(bond.GetIdx())

    return ids_of_bonds_to_cut


def fragment_into_candidate_motifs(
    molecule: Chem.Mol, cut_leaf_edges: bool
) -> List[Tuple[Chem.Mol, List[MotifAtomAnnotation]]]:
    """Fragment a given molecule into candidate motifs.

    The molecule is fragmented on bonds (u, v) such that:
        - (u, v) does not lie on a ring
        - either u or v lies on a ring

    Additionally, if `cut_leaf_edges` is `False`, edges leading to nodes of degree 1 are not cut.

    Returns:
        List of candidate motifs. Each motif is returned as a pair, containing:
            - the corresponding molecular fragment
            - the set of identifiers of nodes in the original molecule corresponding to the motif
    """
    # Copy to make sure the input molecule is unmodified.
    molecule = Chem.Mol(molecule)

    Chem.rdmolops.Kekulize(molecule, clearAromaticFlags=True)

    # Collect identifiers of bridge bonds that will be broken.
    ids_of_bonds_to_cut = get_bonds_to_fragment_on(molecule, cut_leaf_edges=cut_leaf_edges)

    if ids_of_bonds_to_cut:
        # Remove the selected bonds from the molecule.
        fragmented_molecule = Chem.FragmentOnBonds(molecule, ids_of_bonds_to_cut, addDummies=False)
    else:
        fragmented_molecule = molecule

    motifs_as_atom_ids = []
    motifs_as_molecules = Chem.GetMolFrags(
        fragmented_molecule,
        asMols=True,
        sanitizeFrags=False,
        fragsMolAtomMapping=motifs_as_atom_ids,
    )

    # Disable implicit Hs, which interfere with the canonical numbering calculation.
    for atom in fragmented_molecule.GetAtoms():
        atom.SetNoImplicit(True)

    atom_ranks = Chem.CanonicalRankAtoms(fragmented_molecule, breakTies=False)
    atom_annotations: List[List[MotifAtomAnnotation]] = []

    for atom_ids in motifs_as_atom_ids:
        # Convert from a tuple into a sorted list.
        atom_ids = sorted(list(atom_ids))

        # Gather symmetry class ids...
        atom_symmetry_classes = [atom_ranks[atom_id] for atom_id in atom_ids]

        # ...and renumber into [0, 1, ...] for convenience.
        symmetry_classes_present = sorted(list(set(atom_symmetry_classes)))
        atom_symmetry_classes = map(symmetry_classes_present.index, atom_symmetry_classes)

        annotations = [
            MotifAtomAnnotation(atom_id, symmetry_class_id)
            for atom_id, symmetry_class_id in zip(atom_ids, atom_symmetry_classes)
        ]

        atom_annotations.append(list(annotations))

    return list(zip(motifs_as_molecules, atom_annotations))


class MotifVocabularyExtractor:
    def __init__(self, settings: MotifExtractionSettings):
        self._settings = settings
        self._motif_counts = Counter()

    def update(self, molecule: Chem.Mol):
        self._motif_counts.update(
            [
                Chem.MolToSmiles(motif)
                for motif, _ in fragment_into_candidate_motifs(
                    molecule, cut_leaf_edges=self._settings.cut_leaf_edges
                )
            ]
        )

    def output(self):
        motif_list = list(self._motif_counts.items())

        # Sort decreasing by number of occurences, break ties by SMILES string for determinism.
        motif_list = sorted(motif_list, key=lambda element: (element[1], element[0]), reverse=True)

        logger.info(f"Motifs in total: {len(motif_list)}")

        # Filter by minimum frequency if supplied.
        if self._settings.min_frequency is not None:
            motif_list = [
                (motif, frequency)
                for (motif, frequency) in motif_list
                if frequency >= self._settings.min_frequency
            ]

            logger.info(f"Removed motifs occurring less than {self._settings.min_frequency} times")
            logger.info(f"Motifs remaining: {len(motif_list)}")

        motif_list = [
            (motif, frequency, Chem.MolFromSmiles(motif).GetNumAtoms())
            for (motif, frequency) in motif_list
        ]

        motif_list = [
            (motif, frequency, num_atoms)
            for (motif, frequency, num_atoms) in motif_list
            if num_atoms >= self._settings.min_num_atoms
        ]

        logger.info(f"Removing motifs with less than {self._settings.min_num_atoms} atoms")
        logger.info(f"Motifs remaining: {len(motif_list)}")

        # Truncate to maximum vocab size if supplied.
        if self._settings.max_vocab_size is not None:
            motif_list = motif_list[: self._settings.max_vocab_size]

            logger.info(
                f"Truncating the list of motifs to {self._settings.max_vocab_size} most common"
            )
            logger.info(f"Motifs remaining: {len(motif_list)}")

        frequencies = [frequency for (_, frequency, _) in motif_list]
        nums_atoms = [num_atoms for (_, _, num_atoms) in motif_list]

        num_motifs = len(motif_list)

        logger.info("Finished creating the motif vocabulary")
        logger.info(f"| Number of motifs: {num_motifs}")

        if num_motifs > 0:
            logger.info(f"| Min frequency: {min(frequencies)}")
            logger.info(f"| Max frequency: {max(frequencies)}")
            logger.info(f"| Min num atoms: {min(nums_atoms)}")
            logger.info(f"| Max num atoms: {max(nums_atoms)}")

        motif_vocabulary = {
            motif_type: motif_id for motif_id, (motif_type, _, _) in enumerate(motif_list)
        }

        return MotifVocabulary(vocabulary=motif_vocabulary, settings=self._settings)


def find_motifs_from_vocabulary(
    molecule: Chem.Mol, motif_vocabulary: MotifVocabulary
) -> List[MotifAnnotation]:
    """Finds motifs from the vocabulary in a given molecule.

    Args:
        molecule: molecule to find motifs in.
        motif_vocabulary: vocabulary of motifs to recognize.

    Returns:
        List of annotations for all motif occurences found.
    """
    fragments = fragment_into_candidate_motifs(
        molecule, cut_leaf_edges=motif_vocabulary.settings.cut_leaf_edges
    )

    motifs_found = []

    for (motif, atom_annotations) in fragments:
        smiles = Chem.MolToSmiles(motif)

        if smiles in motif_vocabulary.vocabulary:
            motifs_found.append(MotifAnnotation(motif_type=smiles, atoms=atom_annotations))

    return motifs_found


def get_motif_type_to_node_type_index_map(
    motif_vocabulary: MotifVocabulary, num_atom_types: int
) -> Dict[str, int]:
    """Helper to construct a mapping from motif type to shifted node type."""

    return {
        motif: num_atom_types + motif_type
        for motif, motif_type in motif_vocabulary.vocabulary.items()
    }
