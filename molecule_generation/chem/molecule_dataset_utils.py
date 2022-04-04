import logging
import functools
import os
from typing import Any, Dict, Iterable, List, Optional, NamedTuple, Callable, Tuple
import copy

import numpy as np
from dpu_utils.utils import RichPath
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    Descriptors,
    MolFromSmiles,
    rdFingerprintGenerator,
    rdmolops,
    RDConfig,
    MolFromSmarts,
    Mol,
    MolToSmiles,
)
from rdkit.Chem.QED import qed
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import ExactMolWt, BertzCT
from tqdm import tqdm

from molecule_generation.chem.rdkit_helpers import get_atom_symbol
from molecule_generation.chem.motif_utils import (
    find_motifs_from_vocabulary,
    MotifAnnotation,
    MotifExtractionSettings,
    MotifVocabulary,
    MotifVocabularyExtractor,
)
from molecule_generation.chem.atom_feature_utils import (
    AtomFeatureExtractor,
    AtomTypeFeatureExtractor,
    get_default_atom_featurisers,
)
from molecule_generation.utils.sequential_worker_pool import get_worker_pool

logger = logging.getLogger(__name__)

# Note: we do not need to consider aromatic bonds because all molecules have been Kekulized:
# All of the aromatic bonds are converted into either single or double bonds, but the
# "IsAromatic" flag for the bond in unchanged.
BOND_DICT = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2}


class NodeFeatures(NamedTuple):
    """Tuple for holding the return value of `featurise_atoms`."""

    real_valued_features: List[np.ndarray]
    categorical_features: Optional[List[int]]
    num_categorical_classes: Optional[int]


def compute_smiles_dataset_metadata(
    mol_data: Iterable[Dict[str, Any]],
    data_len: Optional[int] = None,
    quiet: bool = False,
    atom_feature_extractors: Optional[List[AtomFeatureExtractor]] = None,
    motif_extraction_settings: Optional[MotifExtractionSettings] = None,
) -> Tuple[List[AtomFeatureExtractor], Optional[MotifVocabulary]]:
    """Given a dataset of molecules, compute metadata (such as atom featuriser vocabularies,
    motif vocabularies).
    """

    if atom_feature_extractors is None:
        uninitialised_featurisers = get_default_atom_featurisers()
        atom_feature_extractors = uninitialised_featurisers
    else:
        uninitialised_featurisers = [
            featuriser
            for featuriser in atom_feature_extractors
            if not featuriser.metadata_initialised
        ]

    if motif_extraction_settings is not None:
        motif_vocabulary_extractor = MotifVocabularyExtractor(motif_extraction_settings)
        logger.info("Initialising feature extractors and motif vocabulary.")
    else:
        motif_vocabulary: Optional[MotifVocabulary] = None
        if len(uninitialised_featurisers) == 0:
            return atom_feature_extractors, motif_vocabulary

        logger.info("Initialising feature extractors.")

    for datapoint in tqdm(mol_data, total=data_len, disable=quiet):
        mol = datapoint["mol"]

        if motif_extraction_settings is not None:
            motif_vocabulary_extractor.update(mol)

        for atom in mol.GetAtoms():
            for featuriser in uninitialised_featurisers:
                featuriser.prepare_metadata(atom)

    if motif_extraction_settings is not None:
        motif_vocabulary = motif_vocabulary_extractor.output()

    for featuriser in uninitialised_featurisers:
        featuriser.mark_metadata_initialised()

    return atom_feature_extractors, motif_vocabulary


def featurise_mol_data(
    mol_data: Iterable[Dict[str, Any]],
    atom_feature_extractors: List[AtomFeatureExtractor],
    motif_vocabulary: Optional[MotifVocabulary] = None,
) -> Iterable[Dict[str, Any]]:
    for datapoint in mol_data:
        try:
            datapoint = dict(datapoint)

            if motif_vocabulary is not None:
                datapoint["motifs"] = find_motifs_from_vocabulary(
                    molecule=datapoint["mol"], motif_vocabulary=motif_vocabulary
                )
            else:
                datapoint["motifs"] = []

            datapoint["graph"] = molecule_to_graph(
                datapoint["mol"],
                atom_feature_extractors,
                motif_vocabulary,
                datapoint["motifs"],
            )

            yield datapoint
        except IndexError:
            print(
                f"Skipping datapoint {datapoint['SMILES']}, cannot featurise with current metadata."
            )
            continue


class FeaturisedData:
    """A tuple to hold the results of featurising a smiles based dataset.

    The class holds four properties about a dataset:
    * atom_feature_extractors: The feature extractors used on the atoms, which
        also store information such as vocabularies used.
    * train_data
    * valid_data
    * test_data
    """

    def __init__(
        self,
        *,
        train_data: Iterable[Dict[str, Any]],
        len_train_data: int,
        valid_data: Iterable[Dict[str, Any]],
        len_valid_data: int,
        test_data: Iterable[Dict[str, Any]],
        len_test_data: int,
        atom_feature_extractors: List[AtomFeatureExtractor],
        featuriser_data: Optional[Iterable[Dict[str, Any]]] = None,
        len_featurizer_data: Optional[int] = None,
        motif_extraction_settings: Optional[MotifExtractionSettings] = None,
        quiet: bool = False,
    ):
        # Store length properties
        self.len_train_data = len_train_data
        self.len_valid_data = len_valid_data
        self.len_test_data = len_test_data

        if featuriser_data is None:
            assert isinstance(
                train_data, list
            ), "If featuriser data is not supplied, then train data must be a list so that it can be iterated over twice."
            featuriser_data = train_data
            len_featurizer_data = len(train_data)

        self._atom_feature_extractors, self._motif_vocabulary = compute_smiles_dataset_metadata(
            mol_data=featuriser_data,
            data_len=len_featurizer_data,
            quiet=quiet,
            atom_feature_extractors=atom_feature_extractors,
            motif_extraction_settings=motif_extraction_settings,
        )

        # Do graph featurisation:
        self._train_data = featurise_mol_data(
            mol_data=train_data,
            atom_feature_extractors=self._atom_feature_extractors,
            motif_vocabulary=self._motif_vocabulary,
        )
        self._valid_data = featurise_mol_data(
            mol_data=valid_data,
            atom_feature_extractors=self._atom_feature_extractors,
            motif_vocabulary=self._motif_vocabulary,
        )
        self._test_data = featurise_mol_data(
            mol_data=test_data,
            atom_feature_extractors=self._atom_feature_extractors,
            motif_vocabulary=self._motif_vocabulary,
        )

    @property
    def train_data(self) -> Iterable[Dict[str, Any]]:
        return self._train_data

    @property
    def valid_data(self) -> Iterable[Dict[str, Any]]:
        return self._valid_data

    @property
    def test_data(self) -> Iterable[Dict[str, Any]]:
        return self._test_data

    @property
    def atom_feature_extractors(self) -> List[AtomFeatureExtractor]:
        return self._atom_feature_extractors

    @property
    def motif_vocabulary(self) -> MotifVocabulary:
        return self._motif_vocabulary


def featurise_smiles_datapoints(
    *,
    train_data: List[Dict[str, Any]],
    valid_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    atom_feature_extractors: List[AtomFeatureExtractor],
    temporary_save_directory: RichPath = None,
    motif_extraction_settings: Optional[MotifExtractionSettings] = None,
    num_processes: int = 8,
    include_fingerprints: bool = False,
    include_descriptors: bool = False,
    include_molecule_stats: bool = True,
    quiet: bool = False,
    filter_failed: bool = False,
) -> FeaturisedData:
    """
    Args:
        train_data: a list of dictionaries representing the training set.
        valid_data: a list of dictionaries representing the validation set.
        test_data: a list of dictionaries representing the test set.
            Note: Each dict must contain a key "SMILES" whose value is a SMILES string
                representing the molecule.
        atom_feature_extractors: list of per-atom feature extractors; graph nodes will
            be labelled by concatenation of their outputs.
        temporary_save_directory: an (optional) directory to cache intermediate results to
            reduce unnecessary recalculation. If used, should be manually cleared if any changes
            have been made to the _smiles_to_mols function.
        num_processes: number of parallel worker processes to use for processing.

    Returns:
        A FeaturisedData tuple.
    """
    tmp_train_path, tmp_test_path, tmp_valid_path = None, None, None
    if temporary_save_directory is not None:
        temporary_save_directory.make_as_dir()
        tmp_train_path = temporary_save_directory.join("train_tmp_feat.pkl.gz")
        tmp_test_path = temporary_save_directory.join("test_tmp_feat.pkl.gz")
        tmp_valid_path = temporary_save_directory.join("valid_tmp_feat.pkl.gz")

    # Step 1: turn smiles into mols:
    logger.info("Turning smiles into mol")
    len_train_data = len(train_data)
    lazy_train_data = _lazy_smiles_to_mols(
        train_data,
        tmp_train_path,
        num_processes,
        include_fingerprints=include_fingerprints,
        include_descriptors=include_descriptors,
        include_molecule_stats=include_molecule_stats,
        filter_failed=filter_failed,
    )
    # Make a copy of the train_data iterator to use in the FeaturisedData class.
    featuriser_data_iter = _lazy_smiles_to_mols(
        train_data,
        tmp_train_path,
        num_processes,
        include_fingerprints=include_fingerprints,
        include_descriptors=include_descriptors,
        include_molecule_stats=include_molecule_stats,
        filter_failed=filter_failed,
    )

    len_valid_data = len(valid_data)
    valid_data_iter = _lazy_smiles_to_mols(
        valid_data,
        tmp_valid_path,
        num_processes,
        include_fingerprints=include_fingerprints,
        include_descriptors=include_descriptors,
        include_molecule_stats=include_molecule_stats,
        filter_failed=filter_failed,
    )
    len_test_data = len(test_data)
    test_data_iter = _lazy_smiles_to_mols(
        test_data,
        tmp_test_path,
        num_processes,
        include_fingerprints=include_fingerprints,
        include_descriptors=include_descriptors,
        include_molecule_stats=include_molecule_stats,
        filter_failed=filter_failed,
    )

    return FeaturisedData(
        train_data=lazy_train_data,
        len_train_data=len_train_data,
        valid_data=valid_data_iter,
        len_valid_data=len_valid_data,
        test_data=test_data_iter,
        len_test_data=len_test_data,
        atom_feature_extractors=atom_feature_extractors,
        featuriser_data=featuriser_data_iter,
        len_featurizer_data=len_train_data,
        motif_extraction_settings=motif_extraction_settings,
        quiet=quiet,
    )


def _need_kekulize(mol):
    """Return whether the given molecule needs to be kekulized."""
    bonds = mol.GetBonds()
    bond_types = [str(bond.GetBondType()) for bond in bonds]
    return any(bond_type == "AROMATIC" for bond_type in bond_types)


def molecule_to_adjacency_lists(mol: Mol) -> List[List[Tuple[int, int]]]:
    """Converts an RDKit molecule to set of list of adjacency lists

    Args:
        mol: the rdkit.ROMol (or RWMol) to be converted.

    Returns:
        A list of lists of edges in the molecule.

    Raises:
        KeyError if there are any aromatic bonds in mol after Kekulization.
    """
    # Kekulize it if needed.
    if _need_kekulize(mol):
        rdmolops.Kekulize(mol)
        if mol is None:
            return None

    # Remove stereo information, such as inward and outward edges
    Chem.RemoveStereochemistry(mol)

    adjacency_lists: List[List[Tuple[int, int]]] = [[] for _ in range(len(BOND_DICT))]
    bonds = mol.GetBonds()
    for bond in bonds:
        bond_type_idx = BOND_DICT[str(bond.GetBondType())]
        adjacency_lists[bond_type_idx].append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    return adjacency_lists


def featurise_atoms(
    mol: Mol,
    atom_feature_extractors: List[AtomFeatureExtractor],
    motif_vocabulary: Optional[MotifVocabulary] = None,
    motifs: List[MotifAnnotation] = [],
) -> NodeFeatures:
    """Computes features (real values, and possibly also categorical) for all atoms.

    Args:
        mol: the molecule to be processed.
        atom_feature_extractors: list of atom feature extractors.
        motif_vocabulary: if running with motifs, a vocabulary of all motif types.
        motifs: if running with motifs, list of motif occurrences in the given molecule.

    Returns:
        NamedTuple, containing node features, and optionally also node classes (i.e. additional node
        features expressed as categorical ids).
    """
    if motif_vocabulary is not None:
        atom_type_feature_extractor = next(
            featuriser
            for featuriser in atom_feature_extractors
            if isinstance(featuriser, AtomTypeFeatureExtractor)
        )

        enclosing_motif_id: Dict[int, int] = {}
        for motif in motifs:
            motif_id = motif_vocabulary.vocabulary[motif.motif_type]

            for atom in motif.atoms:
                enclosing_motif_id[atom.atom_id] = motif_id

        num_motifs = len(motif_vocabulary.vocabulary)

        all_atom_class_ids = []
        num_atom_classes = atom_type_feature_extractor.feature_width + num_motifs
    else:
        assert not motifs

        all_atom_class_ids = None
        num_atom_classes = None

    all_atom_features = []
    for atom_id, atom in enumerate(mol.GetAtoms()):
        atom_symbol = get_atom_symbol(atom)

        atom_features = [
            atom_featuriser.featurise(atom) for atom_featuriser in atom_feature_extractors
        ]

        if motif_vocabulary is not None:
            motif_or_atom_id = enclosing_motif_id.get(
                atom_id, atom_type_feature_extractor.type_name_to_index(atom_symbol) + num_motifs
            )

            assert motif_or_atom_id < num_atom_classes
            all_atom_class_ids.append(motif_or_atom_id)

        atom_features = np.concatenate(atom_features).astype(np.float32)
        all_atom_features.append(atom_features)

    return NodeFeatures(
        real_valued_features=all_atom_features,
        categorical_features=all_atom_class_ids,
        num_categorical_classes=num_atom_classes,
    )


def molecule_to_graph(
    mol: Mol,
    atom_feature_extractors: List[AtomFeatureExtractor],
    motif_vocabulary: Optional[MotifVocabulary] = None,
    motifs: List[MotifAnnotation] = [],
):
    """Converts an RDKit molecule to an encoding of nodes and edges.

    Args:
        mol: the rdkit.ROMol (or RWMol) to be converted.
        atom_feature_extractors: list of per-atom feature extractors; graph nodes will
            be labelled by concatenation of their outputs.

    Returns:
        A dict: {node_labels, node_features, adjacency_list, node_masks)
        node_labels is a string representation of the atom type
        node_features is a vector representation of the atom type.
        adjacency_list is a list of lists of edges in the molecule.

    Raises:
        ValueError if the given molecule cannot be successfully Kekulized.
    """
    if mol is None:
        return None

    # Kekulize it if needed.
    if _need_kekulize(mol):
        rdmolops.Kekulize(mol)
        # Check that there are no aromatic bonds left, fail if there are:
        if _need_kekulize(mol):
            raise ValueError(
                f"Given molecule cannot be Kekulized successfully. "
                f"Molecule has smiles string:\n{MolToSmiles(mol)}"
            )
        if mol is None:
            return None

    # Remove stereo information, such as inward and outward edges
    Chem.RemoveStereochemistry(mol)

    # Calculate the edge information
    adjacency_lists = molecule_to_adjacency_lists(mol)

    graph: Dict[str, List[Any]] = {
        "adjacency_lists": adjacency_lists,
        "node_types": [],
        "node_features": [],
    }

    # Calculate the node information
    for atom in mol.GetAtoms():
        graph["node_types"].append(get_atom_symbol(atom))

    node_features = featurise_atoms(mol, atom_feature_extractors, motif_vocabulary, motifs)

    graph["node_features"] = [
        atom_features.tolist() for atom_features in node_features.real_valued_features
    ]

    if node_features.num_categorical_classes is not None:
        graph["node_categorical_features"] = node_features.categorical_features
        graph["node_categorical_num_classes"] = node_features.num_categorical_classes

    return graph


def _lazy_smiles_to_mols(
    datapoints: Iterable[Dict[str, Any]],
    save_path: RichPath = None,
    num_processes: int = 8,
    include_fingerprints: bool = True,
    include_descriptors: bool = True,
    include_molecule_stats: bool = True,
    report_fail_as_none: bool = False,
    filter_failed: bool = False,
) -> Iterable[Dict[str, Any]]:
    # Early out if we have already done the work:
    if save_path is not None and save_path.exists():
        datapoints = save_path.read_by_file_suffix()
        logger.info(f"Loaded {len(datapoints)} molecules from {save_path}.")
        return datapoints

    # Turn smiles into mols, extract fingerprint data as well:
    with get_worker_pool(num_processes) as p:
        processed_smiles = p.imap(
            functools.partial(
                _smiles_to_rdkit_mol,
                include_fingerprints=include_fingerprints,
                include_descriptors=include_descriptors,
                include_molecule_stats=include_molecule_stats,
                report_fail_as_none=report_fail_as_none or filter_failed,
            ),
            datapoints,
            chunksize=16,
        )

        for processed_datapoint in processed_smiles:
            if filter_failed and processed_datapoint["mol"] is None:
                print("W: Failed to process {} - dropping".format(processed_datapoint["SMILES"]))
            else:
                yield processed_datapoint


def smiles_to_mols(
    datapoints: List[Dict[str, Any]],
    save_path: RichPath = None,
    num_processes: int = 8,
    include_fingerprints: bool = True,
    include_descriptors: bool = True,
    include_molecule_stats: bool = True,
    report_fail_as_none: bool = False,
    quiet: bool = False,
    filter_failed: bool = False,
) -> List[Dict[str, Any]]:
    num_datapoints = len(datapoints)
    datapoints_processed = []
    for result in tqdm(
        _lazy_smiles_to_mols(
            datapoints,
            save_path=save_path,
            num_processes=num_processes,
            include_fingerprints=include_fingerprints,
            include_descriptors=include_descriptors,
            include_molecule_stats=include_molecule_stats,
            report_fail_as_none=report_fail_as_none,
            filter_failed=filter_failed,
        ),
        total=num_datapoints,
        disable=quiet,
    ):
        datapoints_processed.append(result)

    # Save calculated data.
    if save_path is not None:
        save_path.save_as_compressed_file(datapoints_processed)
        logger.info(f"Saved {len(datapoints_processed)} molecules to {save_path}.")
    return datapoints_processed


def get_substructure_match(mol: Mol, scaffold_smarts: str) -> Tuple[int]:
    """Try to find a unique substructure match in the given molecule from the scaffold_smarts string.

    Note: We first try to find a substructure match assuming that the scaffold_smarts is in SMARTS format. If that
    fails to find a match, we try to find one assuming that the given scaffold_smarts is in SMILES format. (Not quite
    the same).

    Returns:
        A tuple of integers, representing the indices of the atoms in the given molecule that correspond to the
        structure defined by the scaffold_smarts.

    Raises:
        ValueError no valid substructure match is found.

    """
    scaffold = MolFromSmarts(scaffold_smarts)
    substructure_matches: Tuple[Tuple[int], ...] = mol.GetSubstructMatches(scaffold)
    if len(substructure_matches) == 0:
        # No match from SMARTS string format. Try SMILES instead.
        smiles_scaffold = MolFromSmiles(scaffold_smarts)
        substructure_matches = mol.GetSubstructMatches(smiles_scaffold)
        if len(substructure_matches) == 0:
            raise ValueError(
                f"No substructure match found for {scaffold_smarts} in molecule {MolToSmiles(mol)}"
            )

    if len(substructure_matches) > 1:
        print(
            f"WARNING: Multiple matches found for {scaffold_smarts} in molecule {MolToSmiles(mol)}. "
            f"Selecting the first."
        )

    return substructure_matches[0]


def _smiles_to_rdkit_mol(
    datapoint,
    include_fingerprints: bool = True,
    include_descriptors: bool = True,
    include_molecule_stats: bool = True,
    report_fail_as_none: bool = False,
) -> Optional[Dict[str, Any]]:
    try:
        smiles_string = datapoint["SMILES"]
        rdkit_mol = MolFromSmiles(smiles_string)

        # copy.deepcopy because calculating features of `rdkit_mol`
        # has side-effects that make it unpickleable.
        # See https://github.com/rdkit/rdkit/issues/3511
        datapoint["mol"] = copy.deepcopy(rdkit_mol)

        # Compute fingerprints:
        if include_fingerprints:
            datapoint["fingerprints_vect"] = rdFingerprintGenerator.GetCountFPs(
                [rdkit_mol], fpType=rdFingerprintGenerator.MorganFP
            )[0]
            fp_numpy = np.zeros((0,), np.int8)  # Generate target pointer to fill
            DataStructs.ConvertToNumpyArray(datapoint["fingerprints_vect"], fp_numpy)
            datapoint["fingerprints"] = fp_numpy

        # Compute descriptors:
        if include_descriptors:
            datapoint["descriptors"] = []
            for descr in Descriptors._descList:
                _, descr_calc_fn = descr
                try:
                    datapoint["descriptors"].append(descr_calc_fn(rdkit_mol))
                except Exception:
                    datapoint["failed_to_convert_from_smiles"] = datapoint["SMILES"]

        # Compute molecule-based scores with RDKit:
        if include_molecule_stats:

            datapoint["properties"] = {
                "sa_score": compute_sa_score(rdkit_mol),
                "clogp": MolLogP(rdkit_mol),
                "mol_weight": ExactMolWt(rdkit_mol),
                "qed": qed(rdkit_mol),
                "bertz": BertzCT(rdkit_mol),
            }

        return datapoint
    except Exception:
        if report_fail_as_none:
            datapoint["mol"] = None
            return datapoint
        else:
            raise


# While the SAScore computation ships with RDKit, it is only in the contrib directory
# and cannot be directly imported. Hence, we need to do a bit of magic to load it,
# and we cache the loaded function in __compute_sascore:
__compute_sascore: Optional[Callable[[Chem.Mol], float]] = None


def compute_sa_score(mol: Chem.Mol, sascorer_path: Optional[str] = None) -> float:
    global __compute_sascore
    if __compute_sascore is None:
        # Guess path to sascorer in RDKit/Contrib if we are not given a path:
        if sascorer_path is None:
            sascorer_path = os.path.join(
                os.path.normpath(RDConfig.RDContribDir), "SA_Score", "sascorer.py"
            )

        # Now import "sascorer.py" by path as a module, and get the core function out:
        import importlib.util

        spec = importlib.util.spec_from_file_location("sascorer", sascorer_path)
        sascorer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sascorer)
        __compute_sascore = sascorer.calculateScore
    return __compute_sascore(mol)
