"""Tests for the CGVAEDecoder class."""
from unittest.mock import patch

import rdkit.Chem
import numpy as np

from molecule_generation.layers.cgvae_decoder import CGVAEDecoder, CGVAEDecoderInput
from molecule_generation.utils.beam_utils import Ray
from molecule_generation.chem.atom_feature_utils import AtomTypeFeatureExtractor
from molecule_generation.preprocessing.cgvae_generation_trace import NodeState


def patched_random_call(self, decoder_input: CGVAEDecoderInput, *_, **__):
    num_edges = decoder_input.valid_edge_choices.shape[0]
    num_edge_types = len(decoder_input.adjacency_lists)

    edge_probabilities = np.random.uniform(size=(num_edges + 1,))
    # Make the stop node less likely.
    edge_probabilities[-1] *= 0.3

    return (np.log(edge_probabilities), np.log(np.random.uniform(size=(num_edges, num_edge_types))))


def patched_always_select_first_call(self, decoder_input: CGVAEDecoderInput, *_, **__):
    num_edges = decoder_input.valid_edge_choices.shape[0]
    num_edge_types = len(decoder_input.adjacency_lists)

    edge_probabilities = np.zeros(shape=(num_edges + 1,)) + 1e-6
    edge_probabilities[0] = 1.0

    edge_type_probabilites = np.zeros(shape=(num_edges, num_edge_types)) + 1e-6
    edge_type_probabilites[:, 0] = 1.0

    return np.log(edge_probabilities), np.log(edge_type_probabilites)


def patch_edge_constraint(valid_edges, *_, **__):
    return valid_edges, None


def patch_edge_type_constraint(valid_edges, adjacency_lists, *_, **__):
    num_edges = valid_edges.shape[0]
    num_edge_types = len(adjacency_lists)
    return np.ones(shape=(num_edges, num_edge_types))


@patch.object(CGVAEDecoder, "__call__", patched_random_call)
def test_one_beam_step_outputs_correct_number_of_rays():
    # Given:
    params = CGVAEDecoder.get_default_params()
    atom_type_featuriser = AtomTypeFeatureExtractor()
    atom_type_featuriser.prepare_metadata(rdkit.Chem.Atom("C"))
    atom_type_featuriser.mark_metadata_initialised()
    decoder = CGVAEDecoder(
        params,
        use_self_loop_edges_in_partial_graphs=False,
        feature_extractors=[atom_type_featuriser],
    )
    decoder._build_distance_embedding_weight()
    num_rays = 5
    num_nodes = 12
    node_types = ["C"] * num_nodes
    embedding_dimension = 100
    num_edge_types = 3
    beam = [
        Ray.construct_ray(
            i,
            focus_node=i,
            num_edge_types=num_edge_types,
            node_types=node_types,
            add_self_loop_edges=False,
        )
        for i in range(num_rays)
    ]
    node_features = np.random.normal(size=(num_nodes, embedding_dimension))
    new_beam = decoder.one_beam_step(
        node_features=node_features,
        node_types=node_types,  # Can be none here because of constraint patches.
        beam=beam,
    )

    # Then:
    assert len(new_beam) == num_rays


@patch.object(CGVAEDecoder, "__call__", patched_always_select_first_call)
def test_one_beam_step_outputs_expected_beam_when_always_selecting_first_edge():
    # Given:
    params = CGVAEDecoder.get_default_params()
    atom_type_featuriser = AtomTypeFeatureExtractor()
    carbon_atom = rdkit.Chem.Atom("C")
    atom_type_featuriser.prepare_metadata(carbon_atom)
    atom_type_featuriser.mark_metadata_initialised()
    decoder = CGVAEDecoder(
        params,
        use_self_loop_edges_in_partial_graphs=False,
        feature_extractors=[atom_type_featuriser],
    )
    decoder._build_distance_embedding_weight()
    num_nodes = 12
    node_types = ["C"] * num_nodes
    embedding_dimension = 100
    num_edge_types = 3
    beam = [
        Ray.construct_ray(
            0,
            focus_node=0,
            num_edge_types=num_edge_types,
            node_types=node_types,
            add_self_loop_edges=False,
        )
    ]
    node_features = np.random.normal(size=(num_nodes, embedding_dimension))
    new_beam = beam

    # What this should do is attach edges between node 0 and 1, 2, 3, 4,
    # and then select the STOP_NODE, moving focus to node 1.
    # Then change the focus node once, so that node 1 becomes locked.
    # And then add edges between node 2 and 3, 4, 5.
    for _ in range(5):
        new_beam = decoder.one_beam_step(
            node_features=node_features,
            node_types=node_types,
            beam=new_beam,
        )
    new_beam[0].update_focus_node()
    for i in range(3):
        new_beam = decoder.one_beam_step(
            node_features=node_features,
            node_types=node_types,
            beam=new_beam,
        )

    # Then:
    expected_node_states = {
        i: NodeState.DISCOVERED if i <= 5 else NodeState.UNDISCOVERED for i in range(num_nodes)
    }
    expected_node_states[
        0
    ] = NodeState.LOCKED  # We've filled up all 4 possible bonds of the first carbon
    expected_node_states[1] = NodeState.LOCKED  # We explicitly moved the focus node once
    expected_node_states[
        2
    ] = NodeState.FOCUS  # This is our current focus, but it now has four bonds as well.

    assert new_beam[0].node_states == expected_node_states

    expected_adjacency_lists = [
        np.zeros(shape=(0, 2), dtype=np.int32) for _ in range(num_edge_types)
    ]
    adjacency_list = np.array(
        [[0, 1], [0, 2], [0, 3], [0, 4], [2, 3], [2, 4], [2, 5]], dtype=np.int32
    )
    reversed_adjacency_list = adjacency_list[:, ::-1]
    # Interleave the adjacency_list and the reversed_adjacency_list:
    symmetrised_adjacency_list = np.empty(
        shape=(2 * len(adjacency_list), 2), dtype=adjacency_list.dtype
    )
    symmetrised_adjacency_list[0::2, :] = adjacency_list
    symmetrised_adjacency_list[1::2, :] = reversed_adjacency_list
    expected_adjacency_lists[0] = symmetrised_adjacency_list

    for calculated_adj_list, expected_adj_list in zip(
        new_beam[0].adjacency_lists, expected_adjacency_lists
    ):
        np.testing.assert_array_equal(calculated_adj_list, expected_adj_list)


@patch.object(CGVAEDecoder, "__call__", patched_random_call)
def test_beam_decode_returns_expected_number_of_rays():
    # Given:
    params = CGVAEDecoder.get_default_params()
    atom_type_featuriser = AtomTypeFeatureExtractor()
    atom_type_featuriser.prepare_metadata(rdkit.Chem.Atom("C"))
    atom_type_featuriser.mark_metadata_initialised()
    decoder = CGVAEDecoder(
        params,
        use_self_loop_edges_in_partial_graphs=False,
        feature_extractors=[atom_type_featuriser],
    )
    decoder._build_distance_embedding_weight()
    num_nodes = 6
    node_types = ["C"] * num_nodes
    embedding_dimension = 100
    beam_size = 5
    node_features = np.random.normal(size=(num_nodes, embedding_dimension))

    # When:
    beams = decoder.beam_decode(
        node_types=node_types, node_features=node_features, beam_size=beam_size
    )

    # Then:
    assert len(beams) == beam_size
