import argparse
from dpu_utils.utils import run_and_debug

from molecule_generation.cli import encode, preprocess, sample, train, visualise
from molecule_generation.utils.cli_utils import setup_logging, supress_tensorflow_warnings


def main():
    parser = argparse.ArgumentParser(
        description="Training, inference and visualization CLI for the MoLeR model."
    )
    subparsers = parser.add_subparsers(help="Subcommand to run.", dest="command", required=True)

    commands = {
        "encode": encode,
        "preprocess": preprocess,
        "sample": sample,
        "train": train,
        "visualise": visualise,
    }
    for command_name, command_module in commands.items():
        inner_parser = command_module.get_argparser()
        subparsers.add_parser(
            command_name,
            parents=[inner_parser],
            description=inner_parser.description,
            add_help=False,
        )

    args = parser.parse_args()

    supress_tensorflow_warnings()
    setup_logging()

    run_and_debug(lambda: commands[args.command].run_from_args(args), getattr(args, "debug", False))


if __name__ == "__main__":
    main()
