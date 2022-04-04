"""Tests for the topology constraint function."""
import numpy as np

import pytest
from rdkit.Chem import Atom, BondType, RWMol

from molecule_generation.chem.topology_features import calculate_topology_features


@pytest.fixture
def molecule():
    """Generating the following molecule:

        0 --- 2 --- 4
        |   / |
        | /   |
        1 --- 3     5

    The edge between 4 and 1 will create a "tri-ring", and all but the edge between 4 and 5 will create a ring.
    """
    mol = RWMol()
    for _ in range(5):
        mol.AddAtom(Atom("C"))
    edges = [[0, 1], [1, 2], [0, 2], [1, 3], [2, 3], [2, 4]]
    for edge in edges:
        mol.AddBond(edge[0], edge[1], BondType.SINGLE)
    return mol


def test_topology_features(molecule):
    # Given:
    # molecule
    potential_edges = np.array([[4, 3], [4, 1], [4, 0], [4, 5]])

    # When:
    edge_features = calculate_topology_features(potential_edges, molecule)

    # Then:
    assert len(edge_features) == len(potential_edges)
    expected_edge_features = np.array([[1, 0], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
    np.testing.assert_array_equal(edge_features, expected_edge_features)
