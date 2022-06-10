import pathlib
import pickle
from typing import Tuple, Union, Dict, Any

from tf2_gnn.cli_utils.model_utils import get_model, get_model_file_path, load_weights_verbosely

from molecule_generation.dataset.in_memory_trace_dataset import InMemoryTraceDataset
from molecule_generation.models.cgvae import CGVAE
from molecule_generation.models.moler_generator import MoLeRGenerator
from molecule_generation.models.moler_vae import MoLeRVae

Pathlike = Union[str, pathlib.Path]


def load_vae_model_and_dataset(
    trained_model_path: Pathlike,
) -> Tuple[InMemoryTraceDataset, Union[CGVAE, MoLeRVae]]:
    trained_model_path = str(
        trained_model_path
    )  # get_model_file_path takes `str` as input, so we need to cast it

    with open(get_model_file_path(trained_model_path, "pkl"), "rb") as in_file:
        data_to_load = pickle.load(in_file)

    atom_type_featuriser = next(
        featuriser
        for featuriser in data_to_load["dataset_metadata"]["feature_extractors"]
        if featuriser.name == "AtomType"
    )

    dummy_dataset = InMemoryTraceDataset(
        params=data_to_load["dataset_params"],
        metadata=data_to_load["dataset_metadata"],
        num_edge_types=data_to_load["num_edge_types"],
        node_feature_shape=data_to_load["node_feature_shape"],
        node_type_index_to_string=atom_type_featuriser.index_to_atom_type_map,
        MoLeR_style_traces=issubclass(data_to_load["model_class"], MoLeRVae),
    )

    model = get_model(
        msg_passing_implementation="does not matter because we supply model_class",
        task_name="does not matter because we supply model_class",
        model_cls=data_to_load["model_class"],
        dataset=dummy_dataset,
        dataset_model_optimised_default_hyperparameters={},
        loaded_model_hyperparameters=data_to_load.get("model_params", {}),
        cli_model_hyperparameter_overrides={},
        hyperdrive_hyperparameter_overrides={},
    )

    data_description = dummy_dataset.get_batch_tf_data_description()
    model.build(data_description.batch_features_shapes)

    weight_file = get_model_file_path(trained_model_path, "hdf5")
    load_weights_verbosely(weight_file, model)

    if not isinstance(model, (CGVAE, MoLeRVae, MoLeRGenerator)):
        raise ValueError(f"Model loaded from {trained_model_path} not a CGVAE or MoLeR model!")

    return dummy_dataset, model


def get_model_parameters(trained_model_path: Pathlike) -> Dict[str, Any]:
    """Returns model parameters from a given pickle path."""
    with open(get_model_file_path(str(trained_model_path), "pkl"), "rb") as in_file:
        # get_model_file_path takes `str` as input, so we need to cast it
        data_to_load = pickle.load(in_file)
    return data_to_load.get("model_params", {})


def get_model_class(trained_model_path: Pathlike) -> type:
    """Returns model class from a given pickle path."""
    with open(get_model_file_path(str(trained_model_path), "pkl"), "rb") as in_file:
        data_to_load = pickle.load(in_file)
    return data_to_load.get("model_class")
