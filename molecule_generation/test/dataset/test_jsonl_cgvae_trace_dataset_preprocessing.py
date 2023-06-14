"""Tests for the jsonl CGVAE trace dataset class."""

import os
import shutil

import pytest
from dpu_utils.utils import RichPath
from tf2_gnn import DataFold

from molecule_generation.dataset.jsonl_cgvae_trace_dataset import JSONLCGVAETraceDataset
from molecule_generation.chem.atom_feature_utils import get_default_atom_featurisers
from molecule_generation.preprocessing.preprocess import preprocess_jsonl_files
from molecule_generation.chem.molecule_dataset_utils import featurise_smiles_datapoints
from molecule_generation.utils.preprocessing_utils import save_data, write_jsonl_gz_data


@pytest.fixture
def interrim_dir():
    save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tmp")
    os.mkdir(save_dir)
    yield save_dir
    # Tear down:
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)


def test_write_smiles_to_jsonl(test_smiles, interrim_dir):
    smiles_dict = [{"SMILES": x.strip()} for x in test_smiles]
    data = featurise_smiles_datapoints(
        train_data=smiles_dict,
        valid_data=smiles_dict,
        test_data=smiles_dict,
        atom_feature_extractors=get_default_atom_featurisers(),
        temporary_save_directory=RichPath.create(interrim_dir),
    )
    save_path_name = os.path.join(interrim_dir, "train.jsonl.gz")
    num_written = write_jsonl_gz_data(save_path_name, data=data.train_data)
    assert num_written == 10


def test_read_and_write_jsonl_files(test_smiles, interrim_dir):
    # Prepare the jsonl.gz files
    smiles_dict = [{"SMILES": x.strip()} for x in test_smiles]
    data = featurise_smiles_datapoints(
        train_data=smiles_dict,
        valid_data=smiles_dict,
        test_data=smiles_dict,
        atom_feature_extractors=get_default_atom_featurisers(),
    )
    save_data(data, output_dir=interrim_dir)

    # Prepare the TraceSample pkl files:
    interrim_rp = RichPath.create(interrim_dir)
    preprocess_jsonl_files(
        jsonl_directory=interrim_rp,
        output_directory=interrim_rp,
        tie_fwd_bkwd_edges=True,
        MoLeR_style_trace=False,
    )

    data_path_template = os.path.join(interrim_dir, "{}_0", "{}_0.pkl.gz")
    for fold in ["train", "valid", "test"]:
        assert os.path.isfile(
            data_path_template.format(fold, fold)
        ), f"File {fold}.pkl.gz does not exist."


@pytest.fixture
def loaded_dataset() -> JSONLCGVAETraceDataset:
    params = JSONLCGVAETraceDataset.get_default_hyperparameters()
    dataset = JSONLCGVAETraceDataset(params)
    data_path = RichPath.create(
        os.path.join(os.path.dirname(__file__), "..", "test_datasets", "cgvae_traces")
    )
    dataset.load_data(data_path, folds_to_load={DataFold.TRAIN, DataFold.VALIDATION, DataFold.TEST})
    yield dataset


def test_num_edge_types(loaded_dataset: JSONLCGVAETraceDataset):
    # We expect 3 tied fwd/bkwd edge types and self loops:
    assert loaded_dataset.num_edge_types == 4


def test_node_feature_shape(loaded_dataset: JSONLCGVAETraceDataset):
    # Fixed in the test dataset:
    assert loaded_dataset.node_feature_shape == (29,)


def test_load_preprocessed_data(loaded_dataset: JSONLCGVAETraceDataset):
    # The TRAIN dataset should repeat 10 items indefinitely. We check it can repeat here:
    train_iterator = loaded_dataset._graph_iterator(DataFold.TRAIN)
    train_iterable = iter(train_iterator)
    num_datapoints = 43
    train_data = [next(train_iterable) for _ in range(num_datapoints)]
    assert len(train_data) == num_datapoints

    # Other two datasets do not repeat.
    assert len(list(loaded_dataset._graph_iterator(DataFold.VALIDATION))) == 10
    assert len(list(loaded_dataset._graph_iterator(DataFold.TEST))) == 10
    train_iterator.cleanup_resources()


def test_batching(loaded_dataset: JSONLCGVAETraceDataset):
    with loaded_dataset.get_context_managed_tf_dataset(DataFold.TEST) as test_data:
        tf_dataset_iterator = iter(test_data.tf_dataset)

        # Test that first minibatch has the right contents:
        first_minibatch = next(tf_dataset_iterator)
        (batch_features, batch_labels) = first_minibatch
        assert len(batch_features.keys()) == 25
        assert "node_features" in batch_features
        assert "node_to_graph_map" in batch_features
        assert "node_to_partial_graph_map" in batch_features
        assert "num_graphs_in_batch" in batch_features
        assert "num_partial_graphs_in_batch" in batch_features
        assert "valid_edge_choices" in batch_features
        assert "partial_node_features" in batch_features
        for edge_type_idx in range(4):
            assert f"adjacency_list_{edge_type_idx}" in batch_features

        assert batch_features["num_graphs_in_batch"] == 10

        assert len(batch_labels.keys()) == 10
        assert "node_types" in batch_labels
        assert "correct_edge_choices" in batch_labels
        assert "num_correct_edge_choices" in batch_labels
        assert "stop_node_label" in batch_labels
        assert "correct_edge_types" in batch_labels
        assert "valid_edge_types" in batch_labels

        try:
            next(tf_dataset_iterator)
            assert False  # iterator should be empty here
        except StopIteration:
            pass  # This is what we expect: The iterator should be finished.
