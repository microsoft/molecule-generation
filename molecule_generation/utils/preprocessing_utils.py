import gzip
import json
import os
import pickle
from collections import Counter, defaultdict
from typing import Iterable, Dict, Optional, Any

import numpy as np

from molecule_generation.chem.molecule_dataset_utils import FeaturisedData
from tqdm import tqdm


def write_jsonl_gz_data(
    file_name: str, data: Iterable[Dict[str, Any]], len_data: int = None
) -> int:
    num_ele = 0
    with gzip.open(file_name, "wt") as data_fh:
        for ele in tqdm(data, total=len_data):
            save_element(ele, data_fh)
            num_ele += 1
    return num_ele


def save_element(element: Dict[str, Any], data_fh) -> None:
    ele = dict(element)
    ele.pop("mol", None)
    ele.pop("fingerprints_vect", None)
    if "fingerprints" in ele:
        ele["fingerprints"] = ele["fingerprints"].tolist()
    data_fh.write(json.dumps(ele) + "\n")


class DataStatistics:
    def __init__(self):
        self._num_atoms_counter = Counter()
        self._atom_type_counter = Counter()
        self._edge_type_counter = Counter()
        self._property_to_all_values = defaultdict(list)

    def update(self, datum):
        self._num_atoms_counter[datum["mol"].GetNumAtoms()] += 1
        self._atom_type_counter.update(datum["graph"]["node_types"])
        for prop_name, prop_value in datum["properties"].items():
            self._property_to_all_values[prop_name].append(prop_value)

    def output(self) -> Dict[str, Any]:
        train_dataset_statistics = {
            "train_atom_num_distribution": self._num_atoms_counter,
            "train_atom_type_distribution": self._atom_type_counter,
        }
        for prop_name, prop_values in self._property_to_all_values.items():
            train_dataset_statistics[f"{prop_name}_mean"] = np.mean(prop_values)
            train_dataset_statistics[f"{prop_name}_stddev"] = np.std(prop_values)
        return train_dataset_statistics


def save_data(featurised_data: FeaturisedData, output_dir: str, quiet: bool = False) -> None:
    os.makedirs(output_dir, exist_ok=True)
    train_data_statistics = DataStatistics()

    for fold_name, data_fold, len_data_fold in zip(
        ["train", "valid", "test"],
        [featurised_data.train_data, featurised_data.valid_data, featurised_data.test_data],
        [
            featurised_data.len_train_data,
            featurised_data.len_valid_data,
            featurised_data.len_test_data,
        ],
    ):
        filename = os.path.join(output_dir, f"{fold_name}.jsonl.gz")
        num_written = 0
        with gzip.open(filename, "wt") as data_fh:
            for datum in tqdm(data_fold, total=len_data_fold, disable=quiet):
                if fold_name == "train":
                    train_data_statistics.update(datum)
                save_element(datum, data_fh)
                num_written += 1

        print(f" Wrote {num_written} datapoints to {filename}.")

    save_metadata(featurised_data, output_dir, extra_metadata=train_data_statistics.output())


def save_metadata(
    featurised_data: FeaturisedData,
    output_dir: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    metadata_file = os.path.join(output_dir, "metadata.pkl.gz")
    if extra_metadata is None:
        extra_metadata = {}
    with gzip.open(metadata_file, "wb") as data_fh:
        pickle.dump(
            {
                **extra_metadata,
                "feature_extractors": featurised_data.atom_feature_extractors,
                "motif_vocabulary": featurised_data.motif_vocabulary,
            },
            data_fh,
        )

    print(f" Wrote metadata to {metadata_file}.")
