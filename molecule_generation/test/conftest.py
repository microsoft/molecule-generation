from pathlib import Path
from typing import List

import pytest


@pytest.fixture
def test_smiles_path() -> str:
    return Path(__file__).resolve().parent / "test_datasets" / "10_test_smiles.smiles"


@pytest.fixture
def test_smiles(test_smiles_path: str) -> List[str]:
    with open(test_smiles_path) as f:
        data = f.readlines()
    return data
