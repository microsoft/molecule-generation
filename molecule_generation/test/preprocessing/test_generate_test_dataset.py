"""Test for the dataset generation script."""
import os
import shutil

import pytest
from dpu_utils.utils import RichPath, LocalPath

from molecule_generation.test.test_datasets.generate_test_dataset import main as generate_main


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


def test_that_folders_and_metadata_get_created_successfully(tmp_test_directory):
    # Run the generation script, with a non-default output directory.
    generate_main(tmp_test_directory)

    # Check for expected file structure:
    fold_names = ["test_0", "train_0", "valid_0"]
    for fold in fold_names:
        data_path = tmp_test_directory.join(fold).join(fold + ".pkl.gz")
        assert data_path.is_file()

    # Check for the metadata:
    metadata_path = tmp_test_directory.join("metadata.pkl.gz")
    assert metadata_path.is_file()
