import pathlib
import random
from typing import ContextManager, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from rdkit import Chem

from molecule_generation.models.moler_generator import MoLeRGenerator
from molecule_generation.models.moler_vae import MoLeRVae
from molecule_generation.utils.moler_decoding_utils import (
    DecoderSamplingMode,
    MoleculeGenerationChoiceInfo,
)
from molecule_generation.utils.model_utils import get_model_class, get_model_parameters


Pathlike = Union[str, pathlib.Path]


class ModelWrapper(ContextManager):
    def __init__(self, dir: Pathlike, seed: int = 0, num_workers: int = 6, beam_size: int = 1):
        # TODO(kmaziarz): Consider whether this should be a `Path` instead.
        self.trained_model_path = str(self._get_model_file(dir))
        self.num_workers = num_workers
        self.beam_size = beam_size

        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        print(f"Loading a trained model from: {self.trained_model_path}")

        # Read latent dimension size. It may have been serialized as str or float, so to be sure we cast it to int.
        raw_latent_repr_size = get_model_parameters(self.trained_model_path)["latent_repr_size"]
        self._latent_size = int(raw_latent_repr_size)

    @classmethod
    def _get_model_file(cls, dir: Pathlike) -> pathlib.Path:
        """Retrieves the MoLeR pickle file from a given directory.

        Args:
            dir: Directory from which the model should be retrieved.

        Returns:
            Path to the model pickle.

        Raises:
            ValueError, if the model pickle is not found or is not unique.
        """
        # Candidate files must end with ".pkl"
        candidates = list(pathlib.Path(dir).glob("*.pkl"))
        if len(candidates) != 1:
            raise ValueError(
                f"There must be exactly one *.pkl file. Found the following: {candidates}."
            )
        else:
            return candidates[0]

    def __enter__(self):
        from molecule_generation.utils.moler_inference_server import MoLeRInferenceServer

        self._inference_server = MoLeRInferenceServer(
            self.trained_model_path,
            num_workers=self.num_workers,
            max_num_samples_per_chunk=500 // self.beam_size,
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):  # type: ignore
        # Standard Python convention, we can ignore the types
        inference_server = getattr(self, "_inference_server", None)
        if inference_server is not None:
            inference_server.__exit__(exc_type, exc_value, traceback)
            delattr(self, "_inference_server")

    def __del__(self):
        inference_server = getattr(self, "_inference_server", None)
        if inference_server is not None:
            inference_server.cleanup_workers()


class VaeWrapper(ModelWrapper):
    """Wrapper for MoLeRVae"""

    def sample_latents(self, num_samples: int) -> List[np.ndarray]:
        """Sample latent vectors from the model's prior.

        Args:
            num_samples: Number of samples to return.

        Returns:
            List of latent vectors.
        """
        return np.random.normal(size=(num_samples, self._latent_size)).astype(np.float32)

    def encode(
        self, smiles_list: List[str], include_log_variances: bool = False
    ) -> Union[List[np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]:
        """Encode input molecules to vectors in the latent space.

        Args:
            smiles_list: List of molecules as SMILES.
            include_log_variances: Whether to also return log variances on the latent encodings.

        Returns:
            List of results. Each result is the mean latent encoding if `include_log_variances` is
            `False`, and a pair containing the mean and the corresponding log variance otherwise.
        """
        # Note: if we ever start being strict about type hints, we could properly express the
        # relationship between `include_log_variances` and the return type using `@overload`.

        return self._inference_server.encode(
            smiles_list, include_log_variances=include_log_variances
        )

    def decode(
        self,
        latents: List[np.ndarray],  # type: ignore
        scaffolds: Optional[List[Optional[str]]] = None,
        include_generation_steps: bool = False,
    ) -> Union[List[str], List[Tuple[str, MoleculeGenerationChoiceInfo]]]:
        """Decode molecules from latent vectors, potentially conditioned on scaffolds.

        Args:
            latents: List of latent vectors to decode.
            scaffolds: List of scaffold molecules, one per each vector. Each scaffold in
                the list can be `None` (denoting lack of scaffold) or the whole list can
                be `None`, which is synonymous with `[None, ..., None]`.
            include_generation_steps: Whether to also track and return various metadata about the
                full generation trace.

        Returns:
            List of results. Each result is just a SMILES string if `include_generation_steps` is
            `False`, and a pair containing the SMILES string and the generation trace otherwise.
        """
        if scaffolds is not None:
            scaffolds = [
                Chem.MolFromSmiles(scaffold) if scaffold is not None else None
                for scaffold in scaffolds
            ]

        results = self._inference_server.decode(
            latent_representations=np.stack(latents),
            include_latent_samples=False,
            include_generation_steps=include_generation_steps,
            init_mols=scaffolds,
            beam_size=self.beam_size,
        )

        if include_generation_steps:
            return [(smiles_str, steps) for smiles_str, _, steps in results]
        else:
            return [smiles_str for smiles_str, _, _ in results]

    def sample(self, num_samples: int) -> List[str]:
        """Sample SMILES strings from the model.

        Args:
            num_samples: Number of samples to return.

        Returns:
            List of SMILES strings.
        """
        return self.decode(self.sample_latents(num_samples))


class GeneratorWrapper(ModelWrapper):
    """Wrapper for MoLeRGenerator model"""

    def sample(self, num_samples: int) -> List[str]:
        latents = np.zeros((num_samples, self._latent_size), dtype=np.float32)
        return [
            smiles_str
            for smiles_str, _ in self._inference_server.decode(
                latent_representations=np.stack(latents),
                include_latent_samples=False,
                init_mols=None,
                beam_size=self.beam_size,
                sampling_mode=DecoderSamplingMode.SAMPLING,
            )
        ]


def load_model_from_directory(model_dir: str, **model_kwargs) -> ModelWrapper:
    model_class_to_wrapper = {MoLeRGenerator: GeneratorWrapper, MoLeRVae: VaeWrapper}
    model_class = get_model_class(ModelWrapper._get_model_file(model_dir))

    if model_class not in model_class_to_wrapper:
        raise ValueError(f"Could not found a wrapper suitable for model class {model_class}")

    return model_class_to_wrapper[model_class](model_dir, **model_kwargs)
