import numpy as np
import pytest
from rdkit.Chem import Mol, RWMol

from molecule_generation.utils.beam_utils import ExtensionType, Ray, RayExtension, extend_beam
from molecule_generation.preprocessing.cgvae_generation_trace import NodeState


@pytest.fixture
def ray():
    return Ray(
        logprob=-0.5,
        idx=0,
        adjacency_lists=[
            np.array([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=np.int32),
            np.array([[0, 2], [2, 0]], dtype=np.int32),
            np.zeros(shape=(0, 2), dtype=np.int32),
        ],
        focus_node=1,
        node_states={
            0: NodeState.LOCKED,
            1: NodeState.FOCUS,
            2: NodeState.DISCOVERED,
            3: NodeState.UNDISCOVERED,
        },
        node_types=["C", "C", "C", "Br"],
        exploration_queue=[2],
        generation_trace=None,
    )


def test_ray_set_focus_node(ray):
    # Given:
    # ray

    # When:
    ray.update_focus_node()

    # Then:
    expected_states = {
        0: NodeState.LOCKED,
        1: NodeState.LOCKED,
        2: NodeState.FOCUS,
        3: NodeState.UNDISCOVERED,
    }
    assert ray.node_states == expected_states
    assert ray.focus_node == 2


def test_ray_add_edge(ray):
    # Given:
    # ray

    # When:
    edge = np.array([1, 3], dtype=np.int32)
    edge_type = 1
    ray.add_edge(edge, edge_type)

    # Then:
    expected_states = {
        0: NodeState.LOCKED,
        1: NodeState.FOCUS,
        2: NodeState.DISCOVERED,
        3: NodeState.DISCOVERED,
    }
    assert ray.node_states == expected_states
    expected_adjacency_lists = [
        np.array([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=np.int32),
        np.array([[0, 2], [2, 0], [1, 3], [3, 1]], dtype=np.int32),
        np.zeros(shape=(0, 2), dtype=np.int32),
    ]
    for calculated_adj_list, expected_adj_list in zip(
        ray.adjacency_lists, expected_adjacency_lists
    ):
        np.testing.assert_array_equal(calculated_adj_list, expected_adj_list)


def test_ray_add_edge_to_empty_adj_list(ray):
    # Given:
    # ray

    # When:
    edge = np.array([1, 3], dtype=np.int32)
    edge_type = 2
    ray.add_edge(edge, edge_type)

    # Then:
    expected_states = {
        0: NodeState.LOCKED,
        1: NodeState.FOCUS,
        2: NodeState.DISCOVERED,
        3: NodeState.DISCOVERED,
    }
    assert ray.node_states == expected_states
    expected_adjacency_lists = [
        np.array([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=np.int32),
        np.array([[0, 2], [2, 0]], dtype=np.int32),
        np.array([[1, 3], [3, 1]], dtype=np.int32),
    ]
    for calculated_adj_list, expected_adj_list in zip(
        ray.adjacency_lists, expected_adjacency_lists
    ):
        np.testing.assert_array_equal(calculated_adj_list, expected_adj_list)


def test_ray_has_rwmol(ray):
    # Given:
    # ray

    # When:
    rwmol = ray.molecule

    # Then:
    assert isinstance(rwmol, RWMol)


def test_ray_has_romol(ray):
    # Given:
    # ray

    # When:
    ro_mol = ray.ro_molecule

    # Then:
    assert isinstance(ro_mol, Mol)


@pytest.fixture
def beam():
    return [
        Ray(
            logprob=-0.5,
            idx=0,
            adjacency_lists=[
                np.array([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=np.int32),
                np.array([[0, 2], [2, 0]], dtype=np.int32),
                np.zeros(shape=(0, 2), dtype=np.int32),
            ],
            focus_node=1,
            node_states={
                0: NodeState.LOCKED,
                1: NodeState.FOCUS,
                2: NodeState.DISCOVERED,
                3: NodeState.UNDISCOVERED,
            },
            node_types=["C", "C", "C", "Br"],
            exploration_queue=[2],
            generation_trace=None,
        ),
        Ray(
            logprob=-0.4,
            idx=1,
            adjacency_lists=[
                np.array([[0, 1], [1, 0]], dtype=np.int32),
                np.zeros(shape=(0, 2), dtype=np.int32),
                np.zeros(shape=(0, 2), dtype=np.int32),
            ],
            focus_node=1,
            node_states={
                0: NodeState.LOCKED,
                1: NodeState.FOCUS,
                2: NodeState.UNDISCOVERED,
                3: NodeState.UNDISCOVERED,
            },
            node_types=["C", "C", "C", "Br"],
            exploration_queue=[],
            generation_trace=None,
        ),
        Ray(
            logprob=-0.6,
            idx=2,
            adjacency_lists=[
                np.array([[0, 1], [1, 0]], dtype=np.int32),
                np.array([[0, 2], [2, 0], [1, 3], [3, 1]], dtype=np.int32),
                np.zeros(shape=(0, 2), dtype=np.int32),
            ],
            focus_node=2,
            node_states={
                0: NodeState.LOCKED,
                1: NodeState.LOCKED,
                2: NodeState.FOCUS,
                3: NodeState.DISCOVERED,
            },
            node_types=["C", "C", "C", "Br"],
            exploration_queue=[3],
            generation_trace=None,
        ),
    ]


def test_extend_beam_with_single_stop_node_choice(beam):
    # Given:
    extension_choice = RayExtension(
        logprob=-0.54,
        edge_choice=None,
        edge_type=None,
        ray_idx=2,
        extension_type=ExtensionType.STOP_NODE,
        generation_step_info=None,
    )

    # When:
    new_beam = extend_beam([extension_choice], beam)

    # Then:
    assert len(new_beam) == 1
    new_ray = new_beam[0]
    assert new_ray.logprob == extension_choice.logprob
    assert new_ray.idx == 0
    # Check the adjacency lists agree:
    chosen_beam = beam[extension_choice.ray_idx]
    for calculated_adj_list, expected_adj_list in zip(
        new_ray.adjacency_lists, chosen_beam.adjacency_lists
    ):
        np.testing.assert_array_equal(calculated_adj_list, expected_adj_list)

    # Check that the focus node has been updated successfully.
    assert new_ray.focus_node == 3  # The only possibility left, given the node states.

    # Check that the node states have been updated successfully.
    expected_node_states = {
        0: NodeState.LOCKED,
        1: NodeState.LOCKED,
        2: NodeState.LOCKED,
        3: NodeState.FOCUS,
    }
    assert new_ray.node_states == expected_node_states


def test_extend_beam_with_single_add_edge_choice(beam):
    # Given:
    extension_choice = RayExtension(
        logprob=-0.54,
        edge_choice=np.array([1, 2], dtype=np.int32),
        edge_type=1,
        ray_idx=1,
        extension_type=ExtensionType.ADD_EDGE,
        generation_step_info=None,
    )

    # When:
    new_beam = extend_beam([extension_choice], beam)

    # Then:
    assert len(new_beam) == 1
    new_ray = new_beam[0]
    assert new_ray.logprob == extension_choice.logprob
    assert new_ray.idx == 0
    # Check the adjacency lists agree:
    expected_adjacency_lists = [
        np.array([[0, 1], [1, 0]], dtype=np.int32),
        np.array([[1, 2], [2, 1]], dtype=np.int32),
        np.zeros(shape=(0, 2), dtype=np.int32),
    ]
    for calculated_adj_list, expected_adj_list in zip(
        new_ray.adjacency_lists, expected_adjacency_lists
    ):
        np.testing.assert_array_equal(calculated_adj_list, expected_adj_list)

    # Check that the node states have been updated successfully.
    expected_node_states = {
        0: NodeState.LOCKED,
        1: NodeState.FOCUS,
        2: NodeState.DISCOVERED,
        3: NodeState.UNDISCOVERED,
    }
    assert new_ray.node_states == expected_node_states


def test_extend_beam_with_two_extension_choices(beam):
    # Given:
    extension_choices = [
        RayExtension(
            logprob=-0.54,
            edge_choice=np.array([1, 2], dtype=np.int32),
            edge_type=1,
            ray_idx=1,
            extension_type=ExtensionType.ADD_EDGE,
            generation_step_info=None,
        ),
        RayExtension(
            logprob=-0.5,
            edge_choice=None,
            edge_type=None,
            ray_idx=0,
            extension_type=ExtensionType.STOP_NODE,
            generation_step_info=None,
        ),
    ]

    # When:
    new_beam = extend_beam(extension_choices, beam)

    # Then:
    assert len(new_beam) == 2

    # Check the beam with the added edge:
    extended_ray = new_beam[0]
    assert extended_ray.logprob == extension_choices[0].logprob
    assert extended_ray.idx == 0
    # Check the adjacency lists agree:
    expected_adjacency_lists = [
        np.array([[0, 1], [1, 0]], dtype=np.int32),
        np.array([[1, 2], [2, 1]], dtype=np.int32),
        np.zeros(shape=(0, 2), dtype=np.int32),
    ]
    for calculated_adj_list, expected_adj_list in zip(
        extended_ray.adjacency_lists, expected_adjacency_lists
    ):
        np.testing.assert_array_equal(calculated_adj_list, expected_adj_list)

    # Check that the node states have been updated successfully.
    expected_node_states = {
        0: NodeState.LOCKED,
        1: NodeState.FOCUS,
        2: NodeState.DISCOVERED,
        3: NodeState.UNDISCOVERED,
    }
    assert extended_ray.node_states == expected_node_states

    # Check the stopped ray:
    stopped_ray = new_beam[1]
    assert stopped_ray.logprob == extension_choices[1].logprob
    assert stopped_ray.idx == 1

    # Check the adjacency lists agree:
    chosen_beam = beam[extension_choices[1].ray_idx]
    for calculated_adj_list, expected_adj_list in zip(
        stopped_ray.adjacency_lists, chosen_beam.adjacency_lists
    ):
        np.testing.assert_array_equal(calculated_adj_list, expected_adj_list)
    expected_node_states = {
        0: NodeState.LOCKED,
        1: NodeState.LOCKED,
        2: NodeState.FOCUS,
        3: NodeState.UNDISCOVERED,
    }
    assert stopped_ray.node_states == expected_node_states
