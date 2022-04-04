"""Tests for the MoLeRDecoder class."""
import pytest
from collections import Counter

import rdkit.Chem
import tensorflow as tf

from molecule_generation.layers.moler_decoder import MoLeRDecoder, MoLeRDecoderInput
from molecule_generation.chem.atom_feature_utils import AtomTypeFeatureExtractor
from molecule_generation.chem.motif_utils import MotifExtractionSettings, MotifVocabulary


@pytest.mark.parametrize("beam_size", [1, 4])
@pytest.mark.parametrize("max_nodes_per_batch", [10, 1000])
def test_returns_correct_number_of_states(beam_size: int, max_nodes_per_batch: int):
    tf.random.set_seed(2)

    # Given:
    atom_type_featuriser = AtomTypeFeatureExtractor()
    atom_type_featuriser.prepare_metadata(rdkit.Chem.Atom("C"))
    atom_type_featuriser.mark_metadata_initialised()

    params = MoLeRDecoder.get_default_params()
    params["max_nodes_per_batch"] = max_nodes_per_batch

    decoder = MoLeRDecoder(
        params=params,
        atom_featurisers=[atom_type_featuriser],
        index_to_node_type_map={0: "UNK", 1: "C", 2: "C1=CC=CC1N"},
        motif_vocabulary=MotifVocabulary(
            vocabulary={"C1=CC=CC1N": 0},
            settings=MotifExtractionSettings(
                min_frequency=None, min_num_atoms=None, cut_leaf_edges=True, max_vocab_size=None
            ),
        ),
    )

    latent_size = 8
    node_features = atom_type_featuriser.feature_width
    edge_features = 3

    decoder.build(
        MoLeRDecoderInput(
            node_features=(None, node_features),
            node_categorical_features=(None,),
            adjacency_lists=tuple((None, 2) for _ in range(4)),
            num_graphs_in_batch=(),
            graph_to_focus_node_map=(None,),
            node_to_graph_map=(None,),
            input_molecule_representations=(None, latent_size),
            graphs_requiring_node_choices=(None,),
            candidate_edges=(None, 2),
            candidate_edge_features=(None, edge_features),
            candidate_attachment_points=(None,),
        )
    )

    num_graphs = 2
    decoder_states = decoder.decode(
        graph_representations=tf.random.normal(shape=(num_graphs, latent_size)),
        beam_size=beam_size,
    )

    # Then:
    assert len(decoder_states) == num_graphs * beam_size

    id_counts = Counter([state.molecule_id for state in decoder_states])
    for id in range(num_graphs):
        assert id_counts[id] == beam_size
