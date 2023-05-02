"""Simple integration test for the data pre-processing pipeline and MoLeR model.

This set of tests:
    * Creates train, valid, test JSONL files from SMIlES strings. Each created dataset will be
      identical, each containing 20 copies of the same molecule.
    * Converts those JSONL files into pre-processed TraceSamples, using the JSONLMoLeRTraceDataset
      pre-processing.
    * Trains the MoLeR model on the pre-processed data for a small number of epochs.
    * Loads the MoLeR weights, runs the model through the test set and checks that the loss gets
      below a (hard-coded) threshold.
"""

import os
import random
import shutil
from typing import List, Optional, Tuple

import numpy as np
import pytest
import tensorflow as tf
from dpu_utils.utils import LocalPath, RichPath
from rdkit import Chem

from tf2_gnn import DataFold
from tf2_gnn.cli_utils.task_utils import register_task
from tf2_gnn.cli_utils.training_utils import (
    get_model_and_dataset,
    get_train_cli_arg_parser,
    load_weights_verbosely,
)

from molecule_generation.dataset.jsonl_moler_trace_dataset import JSONLMoLeRTraceDataset
from molecule_generation.models.moler_vae import MoLeRVae
from molecule_generation.cli.train import run_from_args
from molecule_generation.utils.moler_inference_server import MoLeRInferenceServer
from molecule_generation.chem.atom_feature_utils import get_default_atom_featurisers
from molecule_generation.utils.multiprocessing_utils import run_in_separate_process
from molecule_generation.preprocessing.preprocess import preprocess_jsonl_files
from molecule_generation.chem.molecule_dataset_utils import (
    featurise_smiles_datapoints,
    get_substructure_match,
)
from molecule_generation.utils.preprocessing_utils import save_data

# This represents the chemical with graph:
#
#                 C
#                 |
#     C - C - C - C - Br
#                 |
#                 C
#

SMILES_STRING = "CCC(C)(C)Br"


@pytest.fixture(scope="module")
def tmp_test_directory() -> LocalPath:
    """Create a temporary directory which lives for all of the tests in this module.

    Gets cleaned up after the tests have finished.
    """
    tmp_directory: LocalPath = RichPath.create(os.path.join(os.path.dirname(__file__), "tmp"))
    assert not tmp_directory.exists(), "Tried to create a temporary directory that already exists."
    tmp_directory.make_as_dir()
    yield tmp_directory
    # Tear down.
    if tmp_directory.exists():
        shutil.rmtree(tmp_directory.path, ignore_errors=True)


def get_model_path(tmp_test_directory):
    save_directory = tmp_test_directory.join("save")
    assert save_directory.is_dir()

    model_file_list = save_directory.get_filtered_files_in_dir("*.pkl")
    assert len(model_file_list) == 1
    model_file = model_file_list[0]

    return model_file.path


def test_jsonl_dataset_construction_from_smiles_strings(tmp_test_directory):
    # Construct our dummy dataset of repeated SMILES strings.
    num_elements = 20
    smiles_dict = [{"SMILES": SMILES_STRING} for _ in range(num_elements)]

    # Calculate the featurised data.
    data = featurise_smiles_datapoints(
        train_data=smiles_dict,
        valid_data=smiles_dict,
        test_data=smiles_dict,
        atom_feature_extractors=get_default_atom_featurisers(),
    )

    # Save the featurised data to the jsonl_directory
    jsonl_directory = tmp_test_directory.join("jsonl")
    save_data(data, output_dir=jsonl_directory.path)

    # Check that everything has been created.
    fold_names = ["train", "valid", "test"]
    file_template = "{}.jsonl.gz"
    for fold_name in fold_names:
        assert jsonl_directory.join(file_template.format(fold_name)).is_file()

    assert jsonl_directory.join("metadata.pkl.gz").is_file()


def test_jsonl_to_trace_conversion(tmp_test_directory):
    random.seed(0)
    np.random.seed(0)
    # Do the data conversion.
    jsonl_directory = tmp_test_directory.join("jsonl")
    trace_directory = tmp_test_directory.join("trace")
    trace_directory.make_as_dir()

    preprocess_jsonl_files(
        jsonl_directory=jsonl_directory,
        output_directory=trace_directory,
        tie_fwd_bkwd_edges=True,
        MoLeR_style_trace=True,
    )

    # Make sure that the expected files exist.
    fold_names = ["train_0", "valid_0", "test_0"]
    filename_template = "{}.pkl.gz"
    for fold in fold_names:
        assert trace_directory.join(fold).join(filename_template.format(fold)).is_file()


def test_trace_dataset_properties(tmp_test_directory):
    # Load in the data.
    trace_directory = tmp_test_directory.join("trace")
    params = JSONLMoLeRTraceDataset.get_default_hyperparameters()
    dataset = JSONLMoLeRTraceDataset(params, no_parallelism=True)
    dataset.load_data(trace_directory)

    data_folds = [DataFold.TRAIN, DataFold.TEST, DataFold.VALIDATION]
    expected_num_graphs = 20  # Constant brought forward from above.
    for data_fold in data_folds:
        graph_iterator = dataset._graph_iterator(data_fold)
        graph_iterable = iter(graph_iterator)
        if data_fold == DataFold.TRAIN:
            # Train data is supposed to repeat the dataset infinitely.
            # Check that we extract the expected number of graphs:
            graph_list = [next(graph_iterable) for _ in range(expected_num_graphs)]
            # Now check that we can also extract more graphs:
            next(graph_iterable)
        else:
            # On other folds, the iterator should return each element once:
            graph_list = list(graph_iterator)
            assert len(graph_list) == expected_num_graphs

        # These are hard coded values which depends on the SMILES_STRING constant.
        num_nodes = 6
        num_edges = 5
        num_steps_in_trace = num_nodes + num_edges

        for graph in graph_list:
            assert (
                len(graph.node_types) == num_nodes
            ), "One of the graphs does not have the correct number of nodes."
            assert (
                len(graph.adjacency_lists) == 3
            ), "Unexpected number of edge types encoded in adjacency_lists."
            assert (
                len(graph.adjacency_lists[0])
                == 2 * num_edges  # Factor of 2 because of forward & backward edges.
            ), "One of the graphs does not have the correct number of single bonds."
            assert (
                len(graph.adjacency_lists[1]) == 0
            ), "One of the graphs has at least one double bond."
            assert (
                len(graph.adjacency_lists[2]) == 0
            ), "One of the graphs has at least one triple bond."
            # The following test needs the inequality, because the generation trace gets some steps removed when there
            # are no valid edges. The number of times this happens is random based on the specific generation trace.
            assert (
                len(graph.partial_adjacency_lists) <= num_steps_in_trace
            ), "One of the graphs has a trace that is a not the correct length!"
            assert len(graph.partial_adjacency_lists) == len(
                graph.partial_node_features
            ), "The length of the partial adjacency list and the partial node features don't match."


def test_model_trains_from_data(tmp_test_directory):
    # Register our task:
    register_task(
        task_name="MoLeR",
        dataset_class=JSONLMoLeRTraceDataset,
        dataset_default_hypers={
            "max_nodes_per_batch": 50,
            "max_partial_nodes_per_batch": 50,
            "graph_properties": {},
        },
        model_class=MoLeRVae,
        model_default_hypers={"decoder_num_node_types": 3, "num_train_steps_between_valid": 20},
    )

    # Set up our directory structure.
    trace_directory = tmp_test_directory.join("trace")
    assert trace_directory.is_dir()
    save_directory = tmp_test_directory.join("save")
    # assert not save_directory.exists()
    # save_directory.make_as_dir()

    # Arguments are usually read from the command line.
    parser = get_train_cli_arg_parser()
    arguments: List[str] = [
        "GNN_Edge_MLP",
        "MoLeR",
        trace_directory.path,
        "--save-dir",
        save_directory.path,
        "--max-epochs",
        "5",
        "--seed",
        "0",
    ]
    args = parser.parse_args(arguments)
    # Add arguments which get separately added by our custom test script.
    args.tensorboard = False
    args.profile = False

    # Make TF less noisy:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    tf.get_logger().setLevel("ERROR")

    run_in_separate_process(run_from_args, args)


def test_load_model_weights_and_loss_below_untrained_loss(capsys, tmp_test_directory):
    # Clear the Keras session so that unique naming does not mess up weight loading.
    tf.keras.backend.clear_session()

    # Register our task:
    register_task(
        task_name="MoLeR",
        dataset_class=JSONLMoLeRTraceDataset,
        dataset_default_hypers={
            "max_nodes_per_batch": 50,
            "max_partial_nodes_per_batch": 50,
            "graph_properties": {},
        },
        model_class=MoLeRVae,
        model_default_hypers={"decoder_num_node_types": 3, "num_train_steps_between_valid": 20},
    )

    # Set up our directory structure.
    trace_directory = tmp_test_directory.join("trace")
    assert trace_directory.is_dir()
    save_directory = tmp_test_directory.join("save")
    assert save_directory.is_dir()

    # Arguments are usually read from the command line.
    parser = get_train_cli_arg_parser()
    arguments: List[str] = [
        "GNN_Edge_MLP",
        "MoLeR",
        trace_directory.path,
        "--save-dir",
        save_directory.path,
        "--seed",
        "0",
    ]
    args = parser.parse_args(arguments)

    def get_untrained_and_trained_metrics() -> Optional[Tuple[float, float]]:
        dataset, model = get_model_and_dataset(
            msg_passing_implementation=args.model,
            task_name=args.task,
            data_path=trace_directory,
            trained_model_file=args.load_saved_model,
            cli_data_hyperparameter_overrides=args.data_param_override,
            cli_model_hyperparameter_overrides=args.model_param_override,
            hyperdrive_hyperparameter_overrides={},
            folds_to_load={DataFold.TEST},
        )

        # Calculate test metric for randomly initialised model.
        test_data = dataset.get_tensorflow_dataset(DataFold.TEST)
        _, _, untrained_test_results = model.run_on_data_iterator(
            iter(test_data), training=False, quiet=args.quiet
        )
        untrained_test_metric, _ = model.compute_epoch_metrics(untrained_test_results)

        # Load model weights.
        weight_file_list = save_directory.get_filtered_files_in_dir("*.pkl")

        if len(weight_file_list) != 1:
            return None

        weight_file = weight_file_list[0]
        # Make sure we get no output from the load_weights_verbosely function. Any print statement signals that a weight has
        # not been loaded correctly.
        _ = capsys.readouterr()  # Clear anything printed so far.
        load_weights_verbosely(weight_file.path, model)
        out, _ = capsys.readouterr()  # Capture output from load_weights_verbosely

        if out:
            return None

        _, _, test_results = model.run_on_data_iterator(
            iter(test_data), training=False, quiet=args.quiet
        )
        test_metric, _ = model.compute_epoch_metrics(test_results)

        return untrained_test_metric, test_metric

    result = run_in_separate_process(get_untrained_and_trained_metrics)

    assert result is not None

    untrained_test_metric, test_metric = result
    assert test_metric < untrained_test_metric


def test_can_embed_smiles_with_trained_model(capsys, tmp_test_directory):
    # Clear the Keras session so that unique naming does not mess up weight loading.
    tf.keras.backend.clear_session()

    caffeine_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"

    with MoLeRInferenceServer(get_model_path(tmp_test_directory)) as moler:
        embeddings = moler.encode([caffeine_smiles])

    assert len(embeddings) == 1
    assert len(embeddings[0].shape) == 1


def test_can_decode_smiles_with_trained_model(capsys, tmp_test_directory):
    # Clear the Keras session so that unique naming does not mess up weight loading.
    tf.keras.backend.clear_session()

    caffeine_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"

    with MoLeRInferenceServer(get_model_path(tmp_test_directory)) as moler:
        embedding = moler.encode([caffeine_smiles])
        smiles = [s for s, _, _ in moler.decode(embedding)]

    assert len(smiles) == 1


def test_can_decode_smiles_with_scaffold_trained_model(capsys, tmp_test_directory):
    # Clear the Keras session so that unique naming does not mess up weight loading.
    tf.keras.backend.clear_session()

    base_smiles = "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O"
    scaffold_mol = Chem.MolFromSmiles("CCC1C(O)C(O)C=CC1C(C)NC")  # 14 atoms
    scaffold_smarts = Chem.MolToSmarts(scaffold_mol)
    num_atoms_in_substructure = scaffold_mol.GetNumAtoms()

    with MoLeRInferenceServer(get_model_path(tmp_test_directory)) as moler:
        # Initial Molecules = [Scaffold]
        embedding = moler.encode([base_smiles])
        smiles = [s for s, _, _ in moler.decode(embedding, init_mols=[scaffold_mol])]

    assert len(smiles) == 1

    # Check if the scaffold still exists in all of the samples:
    for sample in smiles:
        mol = Chem.MolFromSmiles(sample)
        substructure_match = get_substructure_match(mol, scaffold_smarts)
        assert len(substructure_match) == num_atoms_in_substructure


def test_can_decode_smiles_list_with_optional_scaffold_trained_model(capsys, tmp_test_directory):
    # Clear the Keras session so that unique naming does not mess up weight loading.
    tf.keras.backend.clear_session()

    base_smiles = "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O"
    scaffold_mol = Chem.MolFromSmiles("CCC1C(O)C(O)C=CC1C(C)NC")  # 14 atoms
    scaffold_smarts = Chem.MolToSmarts(scaffold_mol)
    num_atoms_in_substructure = scaffold_mol.GetNumAtoms()

    with MoLeRInferenceServer(get_model_path(tmp_test_directory)) as moler:
        # Initial Molecules = [Scaffold, None]
        embeddings = moler.encode([base_smiles, base_smiles])
        smiles = [s for s, _, _ in moler.decode(embeddings, init_mols=[scaffold_mol, None])]

        assert len(smiles) == 2

        substructure_match = get_substructure_match(Chem.MolFromSmiles(smiles[0]), scaffold_smarts)
        assert len(substructure_match) == num_atoms_in_substructure

        # Initial Molecules = [None, None]
        embeddings = moler.encode([base_smiles, base_smiles])
        smiles = [s for s, _, _ in moler.decode(embeddings, init_mols=[None, None])]

        assert len(smiles) == 2

        # Initial Molecules = None
        embeddings = moler.encode([base_smiles, base_smiles])
        smiles = [s for s, _, _ in moler.decode(embeddings)]

        assert len(smiles) == 2
