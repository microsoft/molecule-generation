"""Tests for the data_conversion_utils file."""
import pytest
import numpy as np

from rdkit.Chem import BondType, MolToSmiles, CanonSmiles

from molecule_generation.preprocessing.graph_sample import GraphSample, Edge
from molecule_generation.chem.atom_feature_utils import get_default_atom_featurisers
from molecule_generation.preprocessing.data_conversion_utils import (
    convert_graph_sample_to_adjacency_list,
    convert_graph_samples_to_traces,
    _convert_graph_sample_to_graph_trace,
    convert_adjacency_list_to_romol,
)


@pytest.fixture
def graph_sample():
    return GraphSample(
        adjacency_list=[
            Edge(source=0, target=1, type=0),
            Edge(source=1, target=2, type=2),
            Edge(source=2, target=3, type=0),
            Edge(source=3, target=4, type=1),
        ],
        num_edge_types=3,
        node_features=[[0.0, 1.0] for _ in range(5)],
        graph_properties={"sa_score": 1.0},
        node_types=["C" for _ in range(5)],
        smiles_string="some_smiles_string",
    )


@pytest.fixture
def atom_feature_extractors():
    feature_extractors = get_default_atom_featurisers()

    for feature_extractor in feature_extractors:
        # Freeze feature extractors so we can use them right away.
        feature_extractor.mark_metadata_initialised()

    return feature_extractors


def test_convert_graph_sample_to_adjacency_list_with_tie_no_self_loop(graph_sample):
    # Given
    tie_fwd_bkwd_edges = True
    add_self_loops = False
    num_forward_edge_types = 3

    # When
    adjacency_lists = convert_graph_sample_to_adjacency_list(
        graph_sample, num_forward_edge_types, tie_fwd_bkwd_edges, add_self_loops
    )

    # Then
    expected_adjacency_lists = [
        np.array([[0, 1], [2, 3], [1, 0], [3, 2]], dtype=np.int32),
        np.array([[3, 4], [4, 3]], dtype=np.int32),
        np.array([[1, 2], [2, 1]], dtype=np.int32),
    ]
    for calculated_adj_list, expected_adj_list in zip(adjacency_lists, expected_adjacency_lists):
        np.testing.assert_array_equal(calculated_adj_list, expected_adj_list)


def test_convert_graph_sample_to_adjacency_list_with_tie_with_self_loop(graph_sample):
    # Given
    tie_fwd_bkwd_edges = True
    add_self_loops = True
    num_forward_edge_types = 3

    # When
    adjacency_lists = convert_graph_sample_to_adjacency_list(
        graph_sample, num_forward_edge_types, tie_fwd_bkwd_edges, add_self_loops
    )

    # Then
    expected_adjacency_lists = [
        np.array([[0, 1], [2, 3], [1, 0], [3, 2]], dtype=np.int32),
        np.array([[3, 4], [4, 3]], dtype=np.int32),
        np.array([[1, 2], [2, 1]], dtype=np.int32),
        np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.int32),
    ]
    for calculated_adj_list, expected_adj_list in zip(adjacency_lists, expected_adjacency_lists):
        np.testing.assert_array_equal(calculated_adj_list, expected_adj_list)


def test_convert_graph_sample_to_adjacency_list_no_tie_no_self_loop(graph_sample):
    # Given
    tie_fwd_bkwd_edges = False
    add_self_loops = False
    num_forward_edge_types = 3

    # When
    adjacency_lists = convert_graph_sample_to_adjacency_list(
        graph_sample, num_forward_edge_types, tie_fwd_bkwd_edges, add_self_loops
    )

    # Then
    expected_adjacency_lists = [
        np.array([[0, 1], [2, 3]], dtype=np.int32),
        np.array([[3, 4]], dtype=np.int32),
        np.array([[1, 2]], dtype=np.int32),
        np.array([[1, 0], [3, 2]], dtype=np.int32),
        np.array([[4, 3]], dtype=np.int32),
        np.array([[2, 1]], dtype=np.int32),
    ]
    for calculated_adj_list, expected_adj_list in zip(adjacency_lists, expected_adjacency_lists):
        np.testing.assert_array_equal(calculated_adj_list, expected_adj_list)


def test_convert_graph_sample_to_graph_trace_gives_correct_number_of_partial_graphs(
    graph_sample, atom_feature_extractors
):
    # Given:
    num_edge_types = 3
    add_self_loops = False
    tie_fwd_bkwd_edges = True
    num_edges = len(graph_sample.adjacency_list)
    max_node_index = max([max(edge.source, edge.target) for edge in graph_sample.adjacency_list])
    num_nodes = max_node_index + 1

    # When:
    trace_sample = _convert_graph_sample_to_graph_trace(
        graph_sample, num_edge_types, tie_fwd_bkwd_edges, add_self_loops, atom_feature_extractors
    )

    # Then:
    # We will have at most a partial graph for each edge added, and one for each time a focus node changes.
    # The test cannot be a strict equality because steps in the generation trace are removed when there are no valid
    # edge choices. That happens a random number of times depending on the specific route of the generation trace.
    expected_num_partial_graphs = num_edges + num_nodes

    assert len(trace_sample.partial_adjacency_lists) >= num_nodes
    assert len(trace_sample.partial_adjacency_lists) <= expected_num_partial_graphs


def test_convert_graph_sample_to_graph_trace_gives_correct_number_of_partial_properties(
    graph_sample,
    atom_feature_extractors,
):
    # Given:
    num_edge_types = 3
    add_self_loops = False
    tie_fwd_bkwd_edges = True

    # When:
    trace_sample = _convert_graph_sample_to_graph_trace(
        graph_sample, num_edge_types, tie_fwd_bkwd_edges, add_self_loops, atom_feature_extractors
    )
    num_partial_graphs = len(trace_sample.partial_adjacency_lists)

    # Then:
    assert len(trace_sample.correct_edge_choices) == num_partial_graphs
    assert len(trace_sample.valid_edge_choices) == num_partial_graphs
    assert len(trace_sample.correct_edge_types) == num_partial_graphs
    assert len(trace_sample.valid_edge_types) == num_partial_graphs
    assert len(trace_sample.focus_nodes) == num_partial_graphs


def test_convert_graph_samples_to_traces(graph_sample, atom_feature_extractors):
    num_graph_samples = 100
    graph_samples = [graph_sample] * num_graph_samples
    graph_samples = iter(graph_samples)

    trace_iter = convert_graph_samples_to_traces(
        graph_samples, 3, True, False, atom_feature_extractors
    )

    assert len(list(trace_iter)) == num_graph_samples


def test_simple_convert_adjacency_list_to_romol():
    # Given Methyl isocyanate:
    atom_types = ["C", "N", "C", "O"]
    adjacency_lists = [
        np.array([[0, 1]]),
        np.array([[1, 2], [2, 3]]),
    ]
    adjacency_lists_to_bond_type = {0: BondType.SINGLE, 1: BondType.DOUBLE}

    # When:
    mol = convert_adjacency_list_to_romol(
        atom_types=atom_types,
        adjacency_lists=adjacency_lists,
        adjacency_list_to_bond_type=adjacency_lists_to_bond_type,
    )

    # Then:
    expected_smiles_string = "CN=C=O"
    computed_smiles_string = MolToSmiles(mol)
    assert computed_smiles_string == expected_smiles_string


def test_convert_adjacency_list_to_romol():
    # Given:
    caffeine_atom_types = ["C", "N", "C", "N", "C", "C", "C", "O", "N", "C", "O", "N", "C", "C"]
    # Caffeine has 14 heavy atoms in it. Let's make sure we've input the list correctly...
    assert len(caffeine_atom_types) == 14
    fragment_atom_types = ["F", "Cl", "C", "C", "N"]
    atom_types = caffeine_atom_types + fragment_atom_types
    adjacency_lists = [
        np.array(
            [
                [0, 1],
                [1, 2],
                [1, 5],
                [3, 4],
                [5, 6],
                [6, 8],
                [8, 9],
                [8, 13],
                [9, 11],
                [11, 12],
                [11, 4],
                [14, 16],
            ]
        ),
        np.array([[2, 3], [4, 5], [6, 7], [9, 10], [14, 15]]),
    ]
    adjacency_lists_to_bond_type = {0: BondType.SINGLE, 1: BondType.DOUBLE}

    # When:
    mol = convert_adjacency_list_to_romol(
        atom_types=atom_types,
        adjacency_lists=adjacency_lists,
        adjacency_list_to_bond_type=adjacency_lists_to_bond_type,
    )

    # Then:
    # Check the number of bonds:
    expected_num_single_bonds = 11
    expected_num_double_bonds = 4
    num_single_bonds = 0
    num_double_bonds = 0
    total_num_bonds = 0
    for bond in mol.GetBonds():
        total_num_bonds += 1
        bond_type = bond.GetBondType()
        if bond_type == BondType.SINGLE:
            num_single_bonds += 1
        if bond_type == BondType.DOUBLE:
            num_double_bonds += 1
    assert total_num_bonds == expected_num_single_bonds + expected_num_double_bonds
    assert num_single_bonds == expected_num_single_bonds
    assert num_double_bonds == expected_num_double_bonds

    expected_smiles_string = CanonSmiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    computed_smiles_string = CanonSmiles(MolToSmiles(mol))
    assert computed_smiles_string == expected_smiles_string
