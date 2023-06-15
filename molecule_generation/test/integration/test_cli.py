import hashlib
import urllib.request
import pickle
import subprocess
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest


@pytest.fixture(scope="module")
def pretrained_checkpoint_dir() -> Path:
    """Download a pretrained MoLeR checkpoint for testing and remove it afterwards."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        urllib.request.urlretrieve(
            "https://figshare.com/ndownloader/files/34642724", temp_dir_path / "model.pkl"
        )
        yield temp_dir_path


def run_cli(args: List[str]) -> str:
    return subprocess.run(
        ["molecule_generation"] + args, stdout=subprocess.PIPE, check=True, text=True
    ).stdout


def test_encode(pretrained_checkpoint_dir: Path, test_smiles_path: Path) -> None:
    output_path = pretrained_checkpoint_dir / "embeddings.pkl"
    run_cli(["encode", pretrained_checkpoint_dir, str(test_smiles_path), output_path])

    with open(output_path, "rb") as f:
        embeddings = np.stack(pickle.load(f))
    output_path.unlink()

    # There should be one encoding per SMILES in `test_smiles_path`.
    assert embeddings.shape == (10, 512)

    # Compress encodings into their norms and compare with precomputed values.
    expected_norms = np.asarray(
        [4.09043, 3.56717, 5.40588, 5.60358, 5.41453, 5.55465, 3.48990, 4.50119, 4.33559, 5.36916]
    )
    assert np.allclose(np.linalg.norm(embeddings, axis=-1), expected_norms)


def test_sample(pretrained_checkpoint_dir: Path) -> None:
    num_samples = 100
    output = run_cli(["sample", pretrained_checkpoint_dir, str(num_samples)])

    samples = [smiles for smiles in output.split("\n")[1:-1]]
    assert len(samples) == num_samples

    # Check the first three outputs verbatim for easier debugging.
    expected_first_samples = [
        "O=C1C2=CC=C(C3=CC=CC=C3)C=C=C2OC2=CC=CC=C12",
        "CC(=O)NC1=NC2=CC(OCC3=CC=CN(CC4=CC=C(Cl)C=C4)C3=O)=CC=C2N1",
        "CCN1C(=O)C2=CC=CC=C2N=C1NC(C)C(=O)NCC(=O)N=[N+]=[N-]",
    ]
    assert samples[: len(expected_first_samples)] == expected_first_samples

    # Check all samples by comparing to a precomputed hash.
    samples_hash = hashlib.shake_256("\n".join(samples).encode()).hexdigest(16)
    assert samples_hash == "366d78fd2c71c6754a4fd9d403ad8276"
