#!/usr/bin/env python3
import argparse

from molecule_generation.utils.cli_utils import (
    setup_logging,
    supress_tensorflow_warnings,
)
from molecule_generation.visualisation import moler_visualiser_cli, moler_visualiser_html


MODES = {"cli": moler_visualiser_cli, "html": moler_visualiser_html}


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualise molecule generation using a trained model."
    )
    subparsers = parser.add_subparsers(
        help="Output mode for the visualiser.", dest="mode", required=True
    )

    for mode_name, mode_module in MODES.items():
        inner_parser = mode_module.get_argparser()
        subparsers.add_parser(
            mode_name,
            parents=[inner_parser],
            description=inner_parser.description,
            add_help=False,
        )

    return parser


def run_from_args(args: argparse.Namespace) -> None:
    MODES[args.mode].run_from_args(args)


def main() -> None:
    supress_tensorflow_warnings()
    setup_logging()

    run_from_args(get_argparser().parse_args())


if __name__ == "__main__":
    main()
