import tensorflow as tf
import itertools

from rdkit import Chem
from molecule_generation.utils.decoder_batching import batch_decoder_states
from molecule_generation.utils.moler_decoding_utils import MoLeRDecoderState
from molecule_generation.chem.atom_feature_utils import AtomTypeFeatureExtractor
from rdkit.Chem.rdchem import Atom


def do_nothing(*_):
    pass


def test_batch_decoder_states():
    """Check we batch some decoder states into batches in the expected way"""
    decoder_states = [
        MoLeRDecoderState(
            molecule_representation=tf.zeros(3),
            molecule_id=i,
            atom_types=["C"],
            molecule=Chem.MolFromSmiles("C" * i),
        )
        for i in range(1, 100)
    ]
    feature_extractor = AtomTypeFeatureExtractor()
    feature_extractor.prepare_metadata(Atom("C"))
    feature_extractor.mark_metadata_initialised()

    batches = list(
        batch_decoder_states(
            decoder_states=decoder_states,
            atom_featurisers=[feature_extractor],
            motif_vocabulary=None,
            uses_categorical_features=False,
            init_batch_callback=do_nothing,
            add_state_to_batch_callback=do_nothing,
            max_nodes_per_batch=1000,
        )
    )

    # Right number of decoder states returned with each batch
    assert all(
        batch["graphs_in_batch"] == len(decoder_states_batch)
        for batch, decoder_states_batch in batches
    )

    # Check decoder_states were unique, else the test below is not good
    assert len(set(decoder_states)) == len(decoder_states)

    # Check all the decoder states were included in output batches, in the right order
    assert (
        list(
            itertools.chain.from_iterable(
                [decoder_states_batch for _, decoder_states_batch in batches]
            )
        )
        == decoder_states
    )

    assert [batch["graphs_in_batch"] for batch, _ in batches] == [45, 19, 14, 12, 9]
