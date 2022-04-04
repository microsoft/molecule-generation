from collections import Counter
from typing import Set

import numpy as np
import pytest
from molecule_generation.chem.motif_utils import (
    MotifExtractionSettings,
    MotifVocabularyExtractor,
    fragment_into_candidate_motifs,
)
from molecule_generation.chem.atom_feature_utils import get_default_atom_featurisers
from molecule_generation.chem.molecule_dataset_utils import featurise_atoms
from rdkit import Chem


MOLECULE_ETHANOL = Chem.MolFromSmiles("CCO")
MOLECULE_VANILIN = Chem.MolFromSmiles("COC1=C(C=CC(=C1)C=O)O")
MOLECULE_BENZALDEHYDE = Chem.MolFromSmiles("c1ccc(cc1)C=O")


def get_motifs_ethanonl(cut_leaf_edges: bool) -> Set[str]:
    return set(["CCO"])


def get_motifs_vanilin(cut_leaf_edges: bool) -> Set[str]:
    benzene_ring_part_motifs = ["O", "C1=CC=CC=C1"] if cut_leaf_edges else ["OC1=CC=CC=C1"]
    return set(["CO", "C=O"] + benzene_ring_part_motifs)


def get_motifs_benzaldehyde(cut_leaf_edges: bool) -> Set[str]:
    return set(["C1=CC=CC=C1", "C=O"])


@pytest.mark.parametrize("cut_leaf_edges", [False, True])
def test_fragment_into_candidate_motifs(cut_leaf_edges: bool):
    def get_motifs(molecule):
        return set(
            [
                Chem.MolToSmiles(motif)
                for (motif, _) in fragment_into_candidate_motifs(molecule, cut_leaf_edges)
            ]
        )

    assert get_motifs(MOLECULE_ETHANOL) == get_motifs_ethanonl(cut_leaf_edges)
    assert get_motifs(MOLECULE_VANILIN) == get_motifs_vanilin(cut_leaf_edges)
    assert get_motifs(MOLECULE_BENZALDEHYDE) == get_motifs_benzaldehyde(cut_leaf_edges)


@pytest.mark.parametrize("min_frequency", [1, 2, 3])
@pytest.mark.parametrize("min_num_atoms", [1, 6])
@pytest.mark.parametrize("cut_leaf_edges", [False, True])
def test_motif_vocabulary_extractor(min_frequency: int, min_num_atoms: int, cut_leaf_edges: bool):
    extraction_settings = MotifExtractionSettings(
        min_frequency=min_frequency, min_num_atoms=min_num_atoms, cut_leaf_edges=cut_leaf_edges
    )

    extractor = MotifVocabularyExtractor(settings=extraction_settings)

    for molecule in [MOLECULE_VANILIN, MOLECULE_BENZALDEHYDE]:
        extractor.update(molecule)

    result = extractor.output()

    assert result.settings == extraction_settings

    # Compute the expected vocabulary by hand...
    motifs_counter = Counter()
    motifs_counter.update(get_motifs_vanilin(cut_leaf_edges))
    motifs_counter.update(get_motifs_benzaldehyde(cut_leaf_edges))

    motifs_expected = set(
        motif
        for (motif, frequency) in motifs_counter.items()
        if frequency >= min_frequency and Chem.MolFromSmiles(motif).GetNumAtoms() >= min_num_atoms
    )

    # ...and compare to the result from the vocabulary extractor.
    assert result.vocabulary.keys() == motifs_expected
    assert set(result.vocabulary.values()) == set(range(len(motifs_expected)))


@pytest.mark.parametrize("smiles", ["C", "C1C=CC=CC1", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"])
def test_motif_extraction_does_not_change_mol(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    atom_featurisers = get_default_atom_featurisers()

    for featuriser in atom_featurisers:
        for atom in mol.GetAtoms():
            featuriser.prepare_metadata(atom)
        featuriser.mark_metadata_initialised()

    def get_features():
        features = featurise_atoms(
            mol=mol, atom_feature_extractors=atom_featurisers
        ).real_valued_features
        return np.stack(features, axis=0)

    features_before = get_features()

    _ = fragment_into_candidate_motifs(molecule=mol, cut_leaf_edges=True)
    features_after = get_features()

    assert np.allclose(features_before, features_after)
