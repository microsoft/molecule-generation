"""Tests for the valence constraint functions"""

import numpy as np

from molecule_generation.chem.valence_constraints import (
    constrain_edge_choices_based_on_valence,
    constrain_edge_types_based_on_valence,
)


def test_constrain_edge_choices_based_on_valence_on_simple_graph():
    # Given molecule of all carbons:
    #
    #   0 === 1 === 2 --- 3
    #               |
    #               4
    #
    edge_choices = np.array([[3, 0], [3, 1], [3, 4]])
    adjacency_lists = [
        np.array([[2, 4], [4, 2], [2, 3], [3, 2]]),
        np.array([[0, 1], [1, 0], [1, 2], [2, 1]]),
        np.zeros(shape=(0, 2), dtype=np.int32),
    ]
    node_types = ["C"] * 5

    # When:
    target_candidate_mask = constrain_edge_choices_based_on_valence(
        start_node=3,
        candidate_target_nodes=edge_choices[:, 1],
        adjacency_lists=adjacency_lists,
        node_types=node_types,
    )
    constrained_edge_choices = edge_choices[target_candidate_mask]

    # Then:
    expected_edge_choices = np.array([[3, 0], [3, 4]])
    expected_open_edges = np.array([True, False, True])
    np.testing.assert_array_equal(constrained_edge_choices, expected_edge_choices)
    np.testing.assert_array_equal(target_candidate_mask, expected_open_edges)


def test_constrain_edge_choices_based_on_valence_with_empty_edge_choices():
    # Given molecule of all carbons:
    #
    #   0 === 1 === 2 --- 3
    #               |
    #               4
    #
    edge_choices = np.zeros(shape=(0, 2), dtype=np.int32)
    adjacency_lists = [
        np.array([[2, 4], [4, 2], [2, 3], [3, 2]]),
        np.array([[0, 1], [1, 0], [1, 2], [2, 1]]),
        np.zeros(shape=(0, 2), dtype=np.int32),
    ]
    node_types = ["C"] * 5

    # When:
    target_candidate_mask = constrain_edge_choices_based_on_valence(
        start_node=3,
        candidate_target_nodes=edge_choices[:, 1],
        adjacency_lists=adjacency_lists,
        node_types=node_types,
    )

    # Then:
    expected_edge_choices = np.zeros(shape=(0, 2), dtype=np.int32)
    expected_open_edges = np.zeros(shape=(0,), dtype=bool)
    np.testing.assert_array_equal(edge_choices, expected_edge_choices)
    np.testing.assert_array_equal(target_candidate_mask, expected_open_edges)


def test_constrain_edge_choices_based_on_valence_if_focus_node_is_full():
    # Given molecule of all carbons:
    #
    #   0 === 1 === 2 --- 3
    #               |
    #               4
    #
    edge_choices = np.array([[2, 0]])
    adjacency_lists = [
        np.array([[2, 4], [4, 2], [2, 3], [3, 2]]),
        np.array([[0, 1], [1, 0], [1, 2], [2, 1]]),
        np.zeros(shape=(0, 2), dtype=np.int32),
    ]
    node_types = ["C"] * 5

    # When:
    target_candidate_mask = constrain_edge_choices_based_on_valence(
        start_node=2,
        candidate_target_nodes=edge_choices[:, 1],
        adjacency_lists=adjacency_lists,
        node_types=node_types,
    )
    constrained_edge_choices = edge_choices[target_candidate_mask]

    # Then:
    expected_edge_choices = np.zeros(shape=(0, 2), dtype=np.int32)
    expected_open_edges = np.array([False])
    np.testing.assert_array_equal(constrained_edge_choices, expected_edge_choices)
    np.testing.assert_array_equal(target_candidate_mask, expected_open_edges)


def test_constrain_edge_types_based_on_valence_on_simple_graph():
    # Given molecule of all carbons:
    #
    #   0 === 1 === 2 --- 3
    #               |
    #               4
    #
    candidate_targets = np.array([0, 1, 4], dtype=np.int32)
    adjacency_lists = [
        np.array([[2, 4], [4, 2], [2, 3], [3, 2]]),
        np.array([[0, 1], [1, 0], [1, 2], [2, 1]]),
        np.zeros(shape=(0, 2), dtype=np.int32),
    ]
    node_types = ["C"] * 5

    # When:
    edge_type_mask = constrain_edge_types_based_on_valence(
        start_node=3,
        candidate_target_nodes=candidate_targets,
        adjacency_lists=adjacency_lists,
        node_types=node_types,
    )

    # Then:
    expected_edge_type_mask = np.array([[1, 1, 0], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
    np.testing.assert_array_equal(edge_type_mask, expected_edge_type_mask)


def test_constrain_edge_types_based_on_valence_with_empty_edge_choices():
    # Given molecule of all carbons:
    #
    #   0 === 1 === 2 --- 3
    #               |
    #               4
    #
    adjacency_lists = [
        np.array([[2, 4], [4, 2], [2, 3], [3, 2]]),
        np.array([[0, 1], [1, 0], [1, 2], [2, 1]]),
        np.zeros(shape=(0, 2), dtype=np.int32),
    ]
    node_types = ["C"] * 5

    # When:
    edge_type_mask = constrain_edge_types_based_on_valence(
        start_node=3,
        candidate_target_nodes=np.zeros(shape=(0,), dtype=np.int32),
        adjacency_lists=adjacency_lists,
        node_types=node_types,
    )

    # Then:
    expected_edge_type_mask = np.zeros(shape=(0, 3), dtype=np.float64)
    np.testing.assert_array_equal(edge_type_mask, expected_edge_type_mask)


def test_constrain_edge_types_based_on_valence_if_focus_node_is_full():
    # Given molecule of all carbons:
    #
    #   0 === 1 === 2 --- 3
    #               |
    #               4
    #
    candidate_targets = np.array([0], dtype=np.int32)
    adjacency_lists = [
        np.array([[2, 4], [4, 2], [2, 3], [3, 2]]),
        np.array([[0, 1], [1, 0], [1, 2], [2, 1]]),
        np.zeros(shape=(0, 2), dtype=np.int32),
    ]
    node_types = ["C"] * 5

    # When:
    edge_type_mask = constrain_edge_types_based_on_valence(
        start_node=2,
        candidate_target_nodes=candidate_targets,
        adjacency_lists=adjacency_lists,
        node_types=node_types,
    )

    # Then:
    expected_edge_type_mask = np.zeros(shape=(1, 3), dtype=np.float64)
    np.testing.assert_array_equal(edge_type_mask, expected_edge_type_mask)
