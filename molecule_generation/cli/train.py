#!/usr/bin/env python3
"""Trains a given model on a preprocessed dataset."""
import argparse
import random
import json
import os
import time
from typing import Dict, Any, Callable, Tuple, Union

import numpy as np
import tensorflow as tf
import tf2_gnn.cli_utils as cli
import tf2_gnn.cli_utils.training_utils as training_utils
from dpu_utils.utils import RichPath

from molecule_generation.dataset.jsonl_abstract_trace_dataset import JSONLAbstractTraceDataset
from molecule_generation.dataset.jsonl_cgvae_trace_dataset import JSONLCGVAETraceDataset
from molecule_generation.dataset.jsonl_moler_trace_dataset import JSONLMoLeRTraceDataset
from molecule_generation.models.cgvae import CGVAE
from molecule_generation.models.moler_vae import MoLeRVae
from molecule_generation.models.moler_generator import MoLeRGenerator
from molecule_generation.utils.cli_utils import setup_logging, supress_tensorflow_warnings


# Register some tasks, replacing the standard ones:
cli.clear_known_tasks()

dataset_params: Dict[str, Any] = {
    "graph_properties": {
        "sa_score": {
            "type": "regression",
            "graph_representation_size": 64,
            "graph_aggregation_num_heads": 8,
            "graph_aggregation_weighting_fun": "sigmoid",
            "graph_aggregation_hidden_layers": [],
            "predictor_mlp_layer_sizes": [],
            "loss_weight_factor": 0.33,
        },
        "clogp": {
            "type": "regression",
            "graph_representation_size": 64,
            "graph_aggregation_num_heads": 8,
            "graph_aggregation_weighting_fun": "sigmoid",
            "graph_aggregation_hidden_layers": [],
            "predictor_mlp_layer_sizes": [32],
            "loss_weight_factor": 0.33,
        },
        "mol_weight": {
            "type": "regression",
            "graph_representation_size": 16,
            "graph_aggregation_num_heads": 4,
            "graph_aggregation_weighting_fun": "sigmoid",
            "graph_aggregation_hidden_layers": [],
            "predictor_mlp_layer_sizes": [],
            "loss_weight_factor": 0.33,
        },
    }
}

model_debug_hypers = {
    "gnn_hidden_dim": 16,
    "gnn_num_layers": 4,
    "decoder_gnn_hidden_dim": 16,
    "decoder_gnn_num_layers": 4,
    "num_train_steps_between_valid": 10,
}

cli.register_task(
    task_name="CGVAEDebug",
    dataset_class=JSONLCGVAETraceDataset,
    dataset_default_hypers=dataset_params,
    model_class=CGVAE,
    model_default_hypers=model_debug_hypers,
)

dataset_params.update(
    {
        "max_nodes_per_batch": 20000,
        "max_partial_nodes_per_batch": 25000,
    }
)
cli.register_task(
    task_name="CGVAE",
    dataset_class=JSONLCGVAETraceDataset,
    dataset_default_hypers=dataset_params,
    model_class=CGVAE,
    model_default_hypers={"num_train_steps_between_valid": 5000},
)

# Use different dataset configuration for MoLeR, which has a different loss scale
# (everything is roughly 1/14 of the CGVAE losses).
moler_dataset_params = dict(dataset_params)
moler_dataset_params["graph_properties"] = dict(dataset_params.get("graph_properties", {}))
for prop_name, prop_config in dataset_params["graph_properties"].items():
    moler_dataset_params["graph_properties"][prop_name] = dict(prop_config)
    moler_dataset_params["graph_properties"][prop_name]["loss_weight_factor"] = 0.02


cli.register_task(
    task_name="MoLeRDebug",
    dataset_class=JSONLMoLeRTraceDataset,
    dataset_default_hypers=moler_dataset_params,
    model_class=MoLeRVae,
    model_default_hypers=model_debug_hypers,
)


cli.register_task(
    task_name="MoLeR",  # MoLeR VAE
    dataset_class=JSONLMoLeRTraceDataset,
    dataset_default_hypers=moler_dataset_params,
    model_class=MoLeRVae,
    model_default_hypers={"num_train_steps_between_valid": 5000},
)

cli.register_task(
    task_name="MoLeRGenerator",
    dataset_class=JSONLMoLeRTraceDataset,
    dataset_default_hypers=moler_dataset_params,
    model_class=MoLeRGenerator,
    model_default_hypers={"num_train_steps_between_valid": 5000},
)


def run_from_args(args: argparse.Namespace) -> Tuple[str, str, str]:
    # Get the housekeeping going and start logging:
    os.makedirs(args.save_dir, exist_ok=True)
    run_id = training_utils.make_run_id(args.model, args.task)
    log_file = os.path.join(args.save_dir, f"{run_id}.log")

    def log(msg) -> None:
        training_utils.log_line(log_file, msg)

    log(f"Setting random seed {args.random_seed}.")
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)

    data_path = RichPath.create(args.data_path, args.azure_info)
    loaded_model_dataset = training_utils.get_model_and_dataset(
        msg_passing_implementation=args.model,
        task_name=args.task,
        data_path=data_path,
        trained_model_file=args.load_saved_model,
        cli_data_hyperparameter_overrides=args.data_param_override,
        cli_model_hyperparameter_overrides=args.model_param_override,
        folds_to_load={training_utils.DataFold.TRAIN, training_utils.DataFold.VALIDATION},
        load_weights_only=args.load_weights_only,
    )
    if not isinstance(loaded_model_dataset[0], JSONLAbstractTraceDataset):
        raise ValueError(
            f"This training script can only work with TraceDatasets, "
            f"but got {loaded_model_dataset[0]}!"
        )
    dataset: JSONLAbstractTraceDataset = loaded_model_dataset[0]
    if not isinstance(loaded_model_dataset[1], (CGVAE, MoLeRVae, MoLeRGenerator)):
        raise ValueError(
            f"This training script can only work with CGVAE/MoLeR, "
            f"but got {loaded_model_dataset[1]}!"
        )
    model: Union[CGVAE, MoLeRVae, MoLeRGenerator] = loaded_model_dataset[1]

    log(f"Dataset parameters: {json.dumps(training_utils.unwrap_tf_tracked_data(dataset._params))}")
    log(f"Model parameters: {json.dumps(training_utils.unwrap_tf_tracked_data(model._params))}")

    if args.azureml_logging:
        from azureml.core.run import Run

        aml_run = Run.get_context()
    else:
        aml_run = None

    # Set up tensorboard logging.
    if args.tensorboard or args.profile:
        writer = tf.summary.create_file_writer(os.path.join(args.save_dir, "tensorboard"))
        writer.set_as_default()
        tf.summary.experimental.set_step(0)

    trained_model_path = train(
        model,
        dataset,
        log_fun=log,
        run_id=run_id,
        max_epochs=args.max_epochs,
        patience=args.patience,
        save_dir=args.save_dir,
        quiet=args.quiet,
        aml_run=aml_run,
        profile=args.profile,
    )

    if args.run_test and not args.profile:
        data_path = RichPath.create(args.data_path, args.azure_info)
        log("== Running on test dataset")
        log(f"Loading data from {data_path}.")
        dataset.load_data(data_path, {training_utils.DataFold.TEST})
        # Reset the trace keep prob to 1.0, to make test results comparable across
        # different settings. This is required because minibatch size is influenced by
        # the number of kept generation trace steps, and we normalise KL losses based on
        # the number of graphs in a batch.
        orig_keep_prob = dataset._params["trace_element_keep_prob"]
        orig_non_carbon_keep_prob = dataset._params.get("trace_element_non_carbon_keep_prob")
        dataset._params["trace_element_keep_prob"] = 1.0
        dataset._params["trace_element_non_carbon_keep_prob"] = 1.0

        log(f"Restoring best model state from {trained_model_path}.")
        training_utils.load_weights_verbosely(trained_model_path, model)
        try:
            with dataset.get_context_managed_tf_dataset(training_utils.DataFold.TEST) as test_data:
                _, _, test_results = model.run_on_data_iterator(
                    iter(test_data.tf_dataset), training=False, quiet=args.quiet, aml_run=aml_run
                )
                test_metric, test_metric_string = model.compute_epoch_metrics(test_results)
                log(test_metric_string)
                if aml_run is not None:
                    aml_run.log("task_test_metric", float(test_metric))
        finally:
            dataset._params["trace_element_keep_prob"] = orig_keep_prob
            dataset._params["trace_element_non_carbon_keep_prob"] = orig_non_carbon_keep_prob

    return run_id, trained_model_path, log_file


def train(
    model: Union[CGVAE, MoLeRVae, MoLeRGenerator],
    dataset: JSONLAbstractTraceDataset,
    log_fun: Callable[[str], None],
    run_id: str,
    max_epochs: int,
    patience: int,
    save_dir: str,
    quiet: bool = False,
    aml_run=None,
    profile: bool = False,
):
    save_file = os.path.join(save_dir, f"{run_id}_best.pkl")
    num_train_steps_between_valid = 5 if profile else model._params["num_train_steps_between_valid"]
    log_fun(f"Num train steps between valid: {num_train_steps_between_valid}")
    # None here means use the entire valid dataset:
    num_valid_steps = num_train_steps_between_valid if profile else None
    log_fun(f"Num valid steps: {num_valid_steps}")

    # Prepare training data, which essentially generates an endless, repeating dataset,
    # for which we only need a single iterator:
    with dataset.get_context_managed_tf_dataset(training_utils.DataFold.TRAIN) as train_data:
        train_data_iter = iter(train_data.tf_dataset)

        # Prepare validation data, for which we will create a fresh iterator with every use:
        with dataset.get_context_managed_tf_dataset(
            training_utils.DataFold.VALIDATION
        ) as valid_data:
            _, _, initial_valid_results = model.run_on_data_iterator(
                iter(valid_data.tf_dataset),
                training=False,
                quiet=quiet,
                max_num_steps=num_valid_steps,
                aml_run=aml_run,
            )
            best_valid_metric, best_val_str = model.compute_epoch_metrics(initial_valid_results)
            log_fun(f"Initial valid metric: {best_val_str}.")
            training_utils.save_model(save_file, model, dataset, store_weights_in_pkl=True)

            # If profiling, we want only 2 epochs.
            if profile:
                max_epochs = 2

            best_valid_epoch = 0
            train_time_start = time.time()
            for epoch in range(1, max_epochs + 1):
                # Profile the second epoch so we do not see compile times.
                if profile and epoch == 2:
                    tf.profiler.experimental.start(save_dir)

                train_loss, train_speed, train_results = model.run_on_data_iterator(
                    train_data_iter,
                    training=True,
                    quiet=quiet,
                    max_num_steps=num_train_steps_between_valid,
                    aml_run=aml_run,
                )

                if profile and epoch == 2:
                    tf.profiler.experimental.stop(save_dir)

                train_metric, train_metric_string = model.compute_epoch_metrics(train_results)
                log_fun(f"== Results after {epoch * num_train_steps_between_valid} training steps")
                log_fun(
                    f" Train:  {train_loss:.4f} loss | {train_metric_string} | {train_speed:.2f} graphs/s",
                )
                tf.summary.scalar("train_loss", data=train_loss, step=epoch)

                valid_loss, valid_speed, valid_results = model.run_on_data_iterator(
                    iter(valid_data.tf_dataset),
                    training=False,
                    quiet=quiet,
                    max_num_steps=num_valid_steps,
                    aml_run=aml_run,
                )
                tf.summary.scalar("valid_loss", data=valid_loss, step=epoch)

                valid_metric, valid_metric_string = model.compute_epoch_metrics(valid_results)
                log_fun(
                    f" Valid:  {valid_loss:.4f} loss | {valid_metric_string} | {valid_speed:.2f} graphs/s",
                )

                if aml_run is not None:
                    aml_run.log("task_train_metric", float(train_metric))
                    aml_run.log("train_speed", float(train_speed))
                    aml_run.log("task_valid_metric", float(valid_metric))
                    aml_run.log("valid_speed", float(valid_speed))

                # Save if good enough.
                if valid_metric < best_valid_metric:
                    log_fun(
                        f"  (Best results so far, target metric decreased from {best_valid_metric:.5f} to {valid_metric:.5f}.)",
                    )
                    training_utils.save_model(save_file, model, dataset, store_weights_in_pkl=True)
                    best_valid_metric = valid_metric
                    best_valid_epoch = epoch
                elif epoch - best_valid_epoch >= patience:
                    total_time = time.time() - train_time_start
                    log_fun(
                        f"Stopping training after {patience * num_train_steps_between_valid} train steps without "
                        f"improvement on validation metric.",
                    )
                    log_fun(
                        f"Training took {total_time}s. Best validation metric: {best_valid_metric}",
                    )
                    break
    return save_file


def get_argparser() -> argparse.ArgumentParser:
    parser = cli.get_train_cli_arg_parser(default_model_type="GNN_Edge_MLP")

    parser.add_argument(
        "--profile",
        dest="profile",
        help="Make this a profiling run. Warning: will not train the model!",
        action="store_true",
    )

    parser.add_argument(
        "--tensorboard",
        dest="tensorboard",
        help="Log metrics to tensorboard. They will be stored in save_dir/tensorboard.",
        action="store_true",
    )

    # Reduce patience default to reflect the on average substantially larger datasets:
    parser.set_defaults(patience=3)
    return parser


def main() -> None:
    supress_tensorflow_warnings()
    setup_logging()

    run_from_args(get_argparser().parse_args())


if __name__ == "__main__":
    main()
