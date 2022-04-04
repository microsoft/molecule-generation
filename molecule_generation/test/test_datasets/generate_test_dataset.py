"""Generate the test dataset that we use."""

import os
import shutil
import sys
from typing import List

from dpu_utils.utils import RichPath
import logging

from molecule_generation.chem.atom_feature_utils import get_default_atom_featurisers
from molecule_generation.preprocessing.preprocess import preprocess_jsonl_files
from molecule_generation.chem.molecule_dataset_utils import featurise_smiles_datapoints
from molecule_generation.utils.preprocessing_utils import save_data

logger = logging.getLogger(__name__)


def main(output_directory: RichPath = None, moler_style: bool = False):
    data_directory = RichPath.create(os.path.dirname(__file__))
    logger.info(f"Saving data into {data_directory}")
    interrim_dir = data_directory.join("tmp")
    if output_directory is None:
        if moler_style:
            output_directory = data_directory.join("moler_traces")
        else:
            output_directory = data_directory.join("cgvae_traces")

    # Pre-process smiles if needed
    if not interrim_dir.is_dir():
        logger.info("Calculating SMILES.")
        interrim_dir.make_as_dir()
        smiles = _read_smiles(data_directory)
        smiles_dict = [{"SMILES": x.strip()} for x in smiles]
        data = featurise_smiles_datapoints(
            train_data=smiles_dict,
            valid_data=smiles_dict,
            test_data=smiles_dict,
            atom_feature_extractors=get_default_atom_featurisers(),
        )
        save_data(data, output_dir=interrim_dir.path)
    else:
        logger.info(f"Using pre-calculated smiles from {interrim_dir}")

    preprocess_jsonl_files(
        jsonl_directory=interrim_dir,
        output_directory=output_directory,
        tie_fwd_bkwd_edges=True,
        num_processes=1,
        MoLeR_style_trace=moler_style,
    )

    # Clean up if all has run successfully.
    shutil.rmtree(interrim_dir.path, ignore_errors=True)


def _read_smiles(data_directory: RichPath) -> List[str]:
    smiles_file = data_directory.join("10_test_smiles.smiles")
    with open(smiles_file.path) as f:
        data = f.readlines()
    return data


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    import argparse

    parser = argparse.ArgumentParser(description="Generate a test dataset from smiles.")
    parser.add_argument(
        "--output_directory",
        dest="output_directory",
        default=None,
        type=str,
        help="Data output directory.",
    )
    parser.add_argument(
        "--moler",
        dest="moler_style",
        action="store_true",
        help="Switches from generates of CGVAE-style traces to MoLeR-style traces",
    )

    args = parser.parse_args()
    output_directory = args.output_directory
    if output_directory is not None:
        output_directory = RichPath.create(output_directory)

    main(output_directory, moler_style=args.moler_style)
