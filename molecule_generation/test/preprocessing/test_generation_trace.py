"""Tests for the generation trace functionality."""
from unittest.mock import patch

import pytest
from rdkit.Chem import Atom, BondType, RWMol

from molecule_generation.preprocessing.graph_sample import Edge, GraphSample
from molecule_generation.preprocessing.cgvae_generation_trace import (
    graph_sample_to_cgvae_trace,
    calculate_dist_from_focus_to_valid_target,
)


@pytest.fixture
def molecule():
    """Generating the following molecule:

        0 --- 2 --- 4
        |   / |
        | /   |
        1 --- 3

    The edge between atoms 1 and 4 will not be a valid one.
    """
    mol = RWMol()
    for _ in range(5):
        mol.AddAtom(Atom("C"))
    edges = [[0, 1], [1, 2], [0, 2], [1, 3], [2, 3], [2, 4]]
    for edge in edges:
        mol.AddBond(edge[0], edge[1], BondType.SINGLE)
    return mol


def test_calculate_dist_from_focus_to_valid_target():
    """Given the following graph:

             focus
        0 --- 1 --- 4
        | \
        |   \
        2     3     5

    """
    adjacency_list = [
        Edge(source=0, target=1, type=0),
        Edge(source=0, target=2, type=0),
        Edge(source=0, target=3, type=0),
        Edge(source=1, target=4, type=0),
    ]

    valid_edges = [
        Edge(source=1, target=2, type=0),
        Edge(source=1, target=3, type=0),
        Edge(source=1, target=4, type=0),
        Edge(source=1, target=5, type=0),
    ]

    # When:
    distances = calculate_dist_from_focus_to_valid_target(
        adjacency_list=adjacency_list,
        focus_node=1,
        target_nodes=[edge.target for edge in valid_edges],
    )

    # Then:
    assert distances == [2, 2, 1, 0]


def mock_random_choice(*_, **__):
    """Make sure we always start with node 0 as the focus node."""
    return 0


def mock_random_shuffle(*_, **__):
    """Don't do anything when shuffle is called."""
    pass


@patch("random.choice", mock_random_choice)
def test_generation_trace_for_two_nodes():
    # Given:
    simple_graph = GraphSample(
        adjacency_list=[Edge(source=0, target=1, type=0)],
        num_edge_types=3,
        node_features=[0, 1],
        graph_properties={"sa_score": 1.0},
        node_types=["C", "C"],
        smiles_string="CC",
    )

    # When:
    generation_trace = graph_sample_to_cgvae_trace(simple_graph)

    # Then:
    assert generation_trace.focus_nodes == [0, 0, 1]
    assert generation_trace.correct_edge_choices == [[Edge(source=0, target=1, type=0)], [], []]
    assert generation_trace.full_graph == simple_graph

    unconnected_graph = GraphSample(
        adjacency_list=[],
        num_edge_types=3,
        node_features=[0, 1],
        graph_properties={"sa_score": 1.0},
        node_types=["C", "C"],
        smiles_string="CC",
    )
    expected_partial_graphs = [unconnected_graph, simple_graph, simple_graph]
    partial_graphs_without_mol = [
        partial_graph[:-1] + (None,) for partial_graph in generation_trace.partial_graphs
    ]
    assert partial_graphs_without_mol == expected_partial_graphs

    assert generation_trace.valid_edge_choices == [[Edge(source=0, target=1, type=-1)], [], []]


@patch("random.choice", mock_random_choice)
@patch("random.shuffle", mock_random_shuffle)
def test_generation_trace_for_three_nodes():
    # Given:
    adjacency_list = [
        Edge(source=0, target=1, type=0),
        Edge(source=0, target=2, type=0),
        Edge(source=1, target=2, type=1),
    ]
    num_edge_types = 3
    node_features = [0, 1, 0]
    graph_properties = {"sa_score": 1.0}
    node_types = ["C", "C", "F"]
    simple_graph = GraphSample(
        adjacency_list=adjacency_list,
        num_edge_types=num_edge_types,
        node_features=node_features,
        graph_properties=graph_properties,
        node_types=node_types,
        smiles_string="CCF",
    )

    # When:
    generation_trace = graph_sample_to_cgvae_trace(simple_graph)

    # Then:
    assert generation_trace.focus_nodes == [0, 0, 0, 2, 2, 1]
    assert generation_trace.correct_edge_choices == [
        [Edge(source=0, target=1, type=0), Edge(source=0, target=2, type=0)],
        [Edge(source=0, target=1, type=0)],
        [],
        [Edge(source=2, target=1, type=1)],
        [],
        [],
    ]
    assert generation_trace.full_graph == simple_graph

    unconnected_graph = GraphSample(
        adjacency_list=[],
        num_edge_types=num_edge_types,
        node_features=node_features,
        graph_properties=graph_properties,
        node_types=node_types,
        smiles_string="CCF",
    )
    one_edge_graph = GraphSample(
        adjacency_list=[Edge(source=0, target=2, type=0)],
        num_edge_types=num_edge_types,
        node_features=node_features,
        graph_properties=graph_properties,
        node_types=node_types,
        smiles_string="CCF",
    )
    two_edge_graph = GraphSample(
        adjacency_list=[Edge(source=0, target=2, type=0), Edge(source=0, target=1, type=0)],
        num_edge_types=num_edge_types,
        node_features=node_features,
        graph_properties=graph_properties,
        node_types=node_types,
        smiles_string="CCF",
    )
    three_edge_graph = GraphSample(
        adjacency_list=[
            Edge(source=0, target=2, type=0),
            Edge(source=0, target=1, type=0),
            Edge(source=2, target=1, type=1),
        ],
        num_edge_types=num_edge_types,
        node_features=node_features,
        graph_properties=graph_properties,
        node_types=node_types,
        smiles_string="CCF",
    )
    expected_partial_graphs = [
        unconnected_graph,
        one_edge_graph,
        two_edge_graph,
        two_edge_graph,
        three_edge_graph,
        three_edge_graph,
    ]
    partial_graphs_without_mol = [
        partial_graph[:-1] + (None,) for partial_graph in generation_trace.partial_graphs
    ]
    assert partial_graphs_without_mol == expected_partial_graphs

    assert generation_trace.valid_edge_choices == [
        [Edge(source=0, target=1, type=-1), Edge(source=0, target=2, type=-1)],
        [Edge(source=0, target=1, type=-1)],
        [],
        [Edge(source=2, target=1, type=-1)],
        [],
        [],
    ]


@patch("random.choice", mock_random_choice)
@patch("random.shuffle", mock_random_shuffle)
def test_generation_trace_for_four_nodes():
    # Given:
    adjacency_list = [
        Edge(source=0, target=1, type=0),
        Edge(source=0, target=2, type=0),
        Edge(source=1, target=3, type=2),
        Edge(source=3, target=2, type=1),
    ]
    num_edge_types = 3
    node_features = [0, 1, 0, 4]
    graph_properties = {"sa_score": 1.0}
    node_types = ["C", "C", "F", "C"]
    simple_graph = GraphSample(
        adjacency_list=adjacency_list,
        num_edge_types=num_edge_types,
        node_features=node_features,
        graph_properties=graph_properties,
        node_types=node_types,
        smiles_string="CCFC",
    )

    # When:
    generation_trace = graph_sample_to_cgvae_trace(simple_graph)

    # Then:
    assert generation_trace.focus_nodes == [0, 0, 0, 2, 2, 1, 1, 3]
    assert generation_trace.correct_edge_choices == [
        [Edge(source=0, target=1, type=0), Edge(source=0, target=2, type=0)],
        [Edge(source=0, target=1, type=0)],
        [],
        [Edge(source=2, target=3, type=1)],
        [],
        [Edge(source=1, target=3, type=2)],
        [],
        [],
    ]
    full_graph_without_mol = generation_trace.full_graph[:-1] + (None,)
    assert full_graph_without_mol == simple_graph

    assert generation_trace.valid_edge_choices == [
        [
            Edge(source=0, target=1, type=-1),
            Edge(source=0, target=2, type=-1),
            Edge(source=0, target=3, type=-1),
        ],
        [Edge(source=0, target=1, type=-1), Edge(source=0, target=3, type=-1)],
        [Edge(source=0, target=3, type=-1)],
        [Edge(source=2, target=1, type=-1), Edge(source=2, target=3, type=-1)],
        [Edge(source=2, target=1, type=-1)],
        [Edge(source=1, target=3, type=-1)],
        [],
        [],
    ]

    assert generation_trace.distance_to_target == [[0, 0, 0], [0, 0], [0], [2, 0], [2], [3], [], []]
