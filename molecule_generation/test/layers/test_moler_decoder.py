"""Tests for the MoLeRDecoder class."""
import pytest
from collections import Counter
from typing import Any, Dict, List

import rdkit.Chem
import tensorflow as tf

from molecule_generation.chem.atom_feature_utils import AtomTypeFeatureExtractor
from molecule_generation.chem.motif_utils import MotifExtractionSettings, MotifVocabulary
from molecule_generation.layers.moler_decoder import (
    MoLeRDecoder,
    MoLeRDecoderInput,
    MoLeRDecoderState,
)
from molecule_generation.utils.moler_decoding_utils import DecoderSamplingMode


def decode_random_latents(
    num_samples: int, decoder_param_overrides: Dict[str, Any], decode_kwargs: Dict[str, Any]
) -> List[MoLeRDecoderState]:
    tf.random.set_seed(0)

    atom_type_featuriser = AtomTypeFeatureExtractor()
    for atom_type in ["C", "N", "O"]:
        atom_type_featuriser.prepare_metadata(rdkit.Chem.Atom(atom_type))
    atom_type_featuriser.mark_metadata_initialised()

    params = MoLeRDecoder.get_default_params()
    params.update(**decoder_param_overrides)

    decoder = MoLeRDecoder(
        params=params,
        atom_featurisers=[atom_type_featuriser],
        index_to_node_type_map={0: "UNK", 1: "C", 2: "N", 3: "O", 4: "O=[N+][O-]", 5: "C1=CC=CC1N"},
        motif_vocabulary=MotifVocabulary(
            vocabulary={"O=[N+][O-]": 0, "C1=CC=CC1N": 1},
            settings=MotifExtractionSettings(
                min_frequency=None, min_num_atoms=3, cut_leaf_edges=True, max_vocab_size=2
            ),
        ),
    )

    # The latent dimension doesn't matter for correctness; hardcode it to something small.
    latent_dim = 8

    decoder.build(
        MoLeRDecoderInput(
            node_features=(None, atom_type_featuriser.feature_width),
            node_categorical_features=(None,),
            adjacency_lists=tuple((None, 2) for _ in range(4)),
            num_graphs_in_batch=(),
            graph_to_focus_node_map=(None,),
            node_to_graph_map=(None,),
            input_molecule_representations=(None, latent_dim),
            graphs_requiring_node_choices=(None,),
            candidate_edges=(None, 2),
            candidate_edge_features=(None, 3),
            candidate_attachment_points=(None,),
        )
    )

    # Use `SAMPLING` mode (instead of the default `GREEDY`) to cover more different execution paths.
    return decoder.decode(
        graph_representations=tf.random.normal(shape=(num_samples, latent_dim)),
        max_num_steps=10,
        sampling_mode=DecoderSamplingMode.SAMPLING,
        **decode_kwargs,
    )


@pytest.mark.parametrize("beam_size", [1, 3])
@pytest.mark.parametrize("max_nodes_per_batch", [10, 1000])
def test_returns_correct_number_of_states(beam_size: int, max_nodes_per_batch: int):
    num_samples = 2
    decoder_states = decode_random_latents(
        num_samples=num_samples,
        decoder_param_overrides={"max_nodes_per_batch": max_nodes_per_batch},
        decode_kwargs={"beam_size": beam_size},
    )

    assert len(decoder_states) == num_samples * beam_size

    id_counts = Counter([state.molecule_id for state in decoder_states])
    for id in range(num_samples):
        assert id_counts[id] == beam_size


@pytest.mark.parametrize("store_generation_traces", [False, True])
def test_can_store_generation_traces(store_generation_traces: bool):
    decoder_states = decode_random_latents(
        num_samples=10,
        decoder_param_overrides={},
        decode_kwargs={"store_generation_traces": store_generation_traces},
    )

    for decoder_state in decoder_states:
        if store_generation_traces:
            assert decoder_state.generation_steps is not None

            for step in decoder_state.generation_steps:
                # Step metadata classes have a `molecule` field, which is initially set to `None`,
                # and then filled in by the `MoLeRDecoderState` class. Let's make sure all molecules
                # were actually filled in.
                assert step.molecule is not None
        else:
            assert decoder_state.generation_steps is None
