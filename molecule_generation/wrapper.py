import pathlib
import random
from typing import ContextManager, List, Optional, Union

import numpy as np
import tensorflow as tf
from rdkit import Chem


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

        from molecule_generation.utils.model_utils import get_model_parameters

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
        # First, all candidate files must end with "_best.pkl"
        candidates = pathlib.Path(dir).glob("*_best.pkl")
        # Second, the filename (without extension) must match some convention
        candidates = [
            candidate for candidate in candidates if cls._is_moler_model_filename(candidate.stem)
        ]

        if len(candidates) != 1:
            raise ValueError(
                f"There must be exactly one file matching the pattern. Found the following: {candidates}."
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

    def sample_latents(self, num_samples: int) -> List[np.ndarray]:
        """Sample latent vectors from the model's prior.

        Args:
            num_samples: Number of samples to return.

        Returns:
            List of latent vectors.
        """
        return np.random.normal(size=(num_samples, self._latent_size)).astype(np.float32)

    def encode(self, smiles_list: List[str]) -> List[np.ndarray]:
        """Encode input molecules to vectors in the latent space.

        Args:
            smiles_list: List of molecules as SMILES

        Returns:
            List of latent vectors.
        """
        return self._inference_server.encode(smiles_list)

    def decode(
        self,
        latents: List[np.ndarray],  # type: ignore
        scaffolds: Optional[List[Optional[str]]] = None,
    ) -> List[str]:
        """Decode molecules from latent vectors, potentially conditioned on scaffolds.

        Args:
            latents: List of latent vectors to decode.
            scaffolds: List of scaffold molecules, one per each vector. Each scaffold in
                the list can be `None` (denoting lack of scaffold) or the whole list can
                be `None`, which is synonymous with `[None, ..., None]`.

        Returns:
            List of SMILES strings.
        """
        if scaffolds is not None:
            scaffolds = [
                Chem.MolFromSmiles(scaffold) if scaffold is not None else None
                for scaffold in scaffolds
            ]

        return [
            smiles_str
            for smiles_str, _ in self._inference_server.decode(
                latent_representations=np.stack(latents),
                include_latent_samples=False,
                init_mols=scaffolds,
                beam_size=self.beam_size,
            )
        ]

    def sample(self, num_samples: int) -> List[str]:
        """Sample SMILES strings from the model.

        Args:
            num_samples: Number of samples to return.

        Returns:
            List of SMILES strings.
        """
        return self.decode(self.sample_latents(num_samples))

    @staticmethod
    def _is_moler_model_filename(filename: str) -> bool:
        return "_MoLeR__" in filename
