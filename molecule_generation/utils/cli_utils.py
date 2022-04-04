import argparse
import logging
import os
import warnings

import tensorflow as tf


def supress_tensorflow_warnings():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.get_logger().setLevel("ERROR")
    warnings.simplefilter("ignore")


def get_model_loading_parser(
    description: str, include_extra_args: bool = True
) -> argparse.ArgumentParser:
    """Create an `ArgumentParser` relevant to all scripts that load a trained model."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "MODEL_DIR", type=str, help="Directory containing the trained model to use."
    )

    if include_extra_args:
        parser.add_argument("--seed", dest="seed", type=int, help="Random seed to use.")
        parser.add_argument(
            "--num-workers",
            dest="num_workers",
            type=int,
            help="Number of parallel sampling processes.",
        )
        add_debug_flag(parser)

    return parser


def add_debug_flag(parser: argparse.ArgumentParser) -> None:
    """Add a --debug CLI flag to turn on debugging."""
    parser.add_argument("--debug", dest="debug", action="store_true", help="Enable debug routines.")


def setup_logging() -> None:
    format = "%(asctime)s %(filename)s:%(lineno)s %(levelname)s %(message)s"
    logging.basicConfig(format=format, level=logging.INFO)
