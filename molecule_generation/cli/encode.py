#!/usr/bin/env python3
import argparse
import pickle

from molecule_generation import VaeWrapper
from molecule_generation.utils.cli_utils import (
    get_model_loading_parser,
    setup_logging,
    supress_tensorflow_warnings,
)


def save_smiles_embeddings(
    model_dir: str, smiles_path: str, output_path: str, **model_kwargs
) -> None:
    with open(smiles_path, "rt") as f:
        smiles_list = [smiles.rstrip() for smiles in f]

    print(f"Read {len(smiles_list)} SMILES strings.")

    with VaeWrapper(model_dir, **model_kwargs) as model:
        embeddings = model.encode(smiles_list)

    with open(output_path, "wb+") as f:
        pickle.dump(embeddings, f)


def get_argparser() -> argparse.ArgumentParser:
    parser = get_model_loading_parser(description="Encode SMILES strings using a trained model.")
    parser.add_argument(
        "SMILES_PATH", type=str, help="Path to a *.smiles file containing SMILES strings to encode."
    )
    parser.add_argument(
        "OUTPUT_PATH",
        type=str,
        default="embeddings.pkl",
        help="Path to use for the output pickle file.",
    )
    return parser


def run_from_args(args: argparse.Namespace) -> None:
    model_kwargs = {key: getattr(args, key) for key in ["seed", "num_workers"]}
    save_smiles_embeddings(
        args.MODEL_DIR,
        args.SMILES_PATH,
        args.OUTPUT_PATH,
        **{key: value for (key, value) in model_kwargs.items() if value is not None},
    )


def main() -> None:
    supress_tensorflow_warnings()
    setup_logging()

    run_from_args(get_argparser().parse_args())


if __name__ == "__main__":
    main()
