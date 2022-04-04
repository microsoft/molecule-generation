"""Tests for the CGVAE class."""
import os
import random

import numpy as np
import pytest
import tensorflow as tf
from dpu_utils.utils import RichPath
from tf2_gnn import DataFold

from molecule_generation.dataset.jsonl_cgvae_trace_dataset import JSONLCGVAETraceDataset
from molecule_generation.models.cgvae import CGVAE


@pytest.fixture
def dataset():
    dataset_params = JSONLCGVAETraceDataset.get_default_hyperparameters()
    dataset_params["graph_properties"] = {
        "sa_score": {
            "type": "regression",
            "loss_weight_factor": 1.0,
        },
    }
    dataset_params.update(
        {
            "max_nodes_per_batch": 50,
            "max_partial_nodes_per_batch": 50,
        }
    )
    dataset = JSONLCGVAETraceDataset(dataset_params)
    data_path = RichPath.create(
        os.path.join(os.path.dirname(__file__), "..", "test_datasets", "cgvae_traces")
    )
    dataset.load_data(data_path, folds_to_load={DataFold.TRAIN, DataFold.VALIDATION})

    return dataset


def test_train_improvement(dataset):
    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)

    # Turn off warnings in TF model construction, which are expected noise:
    def ignore_warn(*args, **kwargs):
        pass

    import warnings

    warnings.warn = ignore_warn

    model_params = CGVAE.get_default_hyperparameters()
    model = CGVAE(
        model_params,
        dataset=dataset,
    )
    data_description = dataset.get_batch_tf_data_description()
    model.build(data_description.batch_features_shapes)

    # We run once on validation, do one training epoch, and then assert that results have improved:
    with dataset.get_context_managed_tf_dataset(DataFold.TRAIN) as train_data:
        with dataset.get_context_managed_tf_dataset(DataFold.VALIDATION) as valid_data:
            train_data_iter = iter(train_data.tf_dataset)  # Re-use endless iterator

            valid0_loss, _, valid0_results = model.run_on_data_iterator(
                data_iterator=iter(valid_data.tf_dataset), training=False, quiet=True
            )
            valid0_metric, _ = model.compute_epoch_metrics(valid0_results)

            train1_loss, _, train1_results = model.run_on_data_iterator(
                data_iterator=train_data_iter,
                training=True,
                quiet=True,
                max_num_steps=20,
            )
            train1_metric, _ = model.compute_epoch_metrics(train1_results)

            valid1_loss, _, valid1_results = model.run_on_data_iterator(
                data_iterator=iter(valid_data.tf_dataset),
                training=False,
                quiet=True,
            )
            valid1_metric, _ = model.compute_epoch_metrics(valid1_results)

            assert valid0_loss > valid1_loss
            assert valid0_metric > valid1_metric

            train2_loss, _, train2_results = model.run_on_data_iterator(
                data_iterator=train_data_iter,
                training=True,
                quiet=True,
                max_num_steps=20,
            )
            train2_metric, _ = model.compute_epoch_metrics(train2_results)

            assert train1_loss > train2_loss
            assert train1_metric > train2_metric
