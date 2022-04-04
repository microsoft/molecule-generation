from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict

import numpy as np
from rdkit.Chem.rdchem import Atom

from molecule_generation.chem.rdkit_helpers import get_atom_symbol


class FeatureType(Enum):
    CategoryId = 1
    BoolValue = 2
    PositiveNumber = 3
    Other = 4


class AtomFeatureExtractor(ABC):
    """
    Abstract parent class of all atom-wise feature extractors.
    Feature extractors are first provided with all training data (to build up
    vocabularies and similar things), and can then be applied to featurise an atom.
    """

    def __init__(self, name: str):
        self._name = name
        self._metadata_initialised = False

    @property
    def name(self) -> str:
        return self._name

    def prepare_metadata(self, atom: Atom) -> None:
        pass

    @property
    @abstractmethod
    def feature_type(self) -> FeatureType:
        raise NotImplementedError()

    @property
    @abstractmethod
    def feature_width(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def featurise(self, atom: Atom) -> np.ndarray:
        raise NotImplementedError()

    @property
    def metadata_initialised(self) -> bool:
        return self._metadata_initialised

    def _assert_metadata_uninitialised(self) -> None:
        if self.metadata_initialised:
            raise ValueError(f"Trying to modify metadata of FeatureExtractor with frozen metadata.")

    def mark_metadata_initialised(self) -> None:
        self._metadata_initialised = True

    def _assert_metadata_initialised(self) -> None:
        if not self.metadata_initialised:
            raise ValueError(f"Trying to use FeatureExtractor without frozen metadata.")

    @property
    def masked_features(self) -> np.ndarray:
        features = np.zeros(shape=(self.feature_width,))
        # For categorical feature encodings, we use the first entry as "unknown"
        # by convention; everything else is just set to 0:
        if self.feature_type == FeatureType.CategoryId:
            features[0] = 1.0
        # For boolean features, we set the value to 0.5:
        if self.feature_type == FeatureType.BoolValue:
            features[0] = 0.5
        return features


class AtomTypeFeatureExtractor(AtomFeatureExtractor):
    def __init__(self):
        super().__init__("AtomType")
        self._atom_type_vocabulary = {"UNK": 0}
        self._index_to_atom_type: Dict[int, str] = {}

    def prepare_metadata(self, atom: Atom) -> None:
        self._assert_metadata_uninitialised()
        atom_symbol = get_atom_symbol(atom)
        if atom_symbol not in self._atom_type_vocabulary:
            self._atom_type_vocabulary[atom_symbol] = len(self._atom_type_vocabulary)

    def mark_metadata_initialised(self) -> None:
        for k, v in self._atom_type_vocabulary.items():
            self._index_to_atom_type[v] = k
        super().mark_metadata_initialised()

    @property
    def feature_type(self) -> FeatureType:
        return FeatureType.CategoryId

    @property
    def feature_width(self) -> int:
        return len(self._atom_type_vocabulary)

    @property
    def index_to_atom_type_map(self) -> Dict[int, str]:
        self._assert_metadata_initialised()
        return self._index_to_atom_type

    def type_name_to_index(self, type_name: str) -> int:
        return self._atom_type_vocabulary.get(type_name, 0)

    def featurise_type_name(self, type_name: str) -> np.ndarray:
        self._assert_metadata_initialised()
        features = np.zeros(shape=(self.feature_width,))
        features[self.type_name_to_index(type_name)] = 1.0
        return features

    def featurise(self, atom: Atom) -> np.ndarray:
        type_name = get_atom_symbol(atom)
        return self.featurise_type_name(type_name)


class AtomDegreeFeatureExtractor(AtomFeatureExtractor):
    def __init__(self, encode_as_onehot: bool = False):
        super().__init__("Degree")
        self._encode_as_onehot = encode_as_onehot
        self._min_known_degree = 1
        self._max_known_degree = 1

    def prepare_metadata(self, atom: Atom) -> None:
        self._assert_metadata_uninitialised()
        self._min_known_degree = min(self._min_known_degree, atom.GetDegree())
        self._max_known_degree = max(self._max_known_degree, atom.GetDegree())

    @property
    def feature_type(self) -> FeatureType:
        if self._encode_as_onehot:
            return FeatureType.CategoryId
        else:
            return FeatureType.PositiveNumber

    @property
    def feature_width(self) -> int:
        if self._encode_as_onehot:
            # Note that we use the first entry of the one-hot vector to signify "unknown"
            return 1 + self._max_known_degree - self._min_known_degree + 1
        else:
            return 1

    def featurise(self, atom: Atom) -> np.ndarray:
        self._assert_metadata_initialised()
        if self._encode_as_onehot:
            features = np.zeros(shape=(self.feature_width,))
            features[1 + atom.GetDegree() - self._min_known_degree] = 1.0
            return features
        else:
            return np.array([atom.GetDegree()])


class AtomChargeFeatureExtractor(AtomFeatureExtractor):
    def __init__(self, encode_as_onehot: bool = False):
        super().__init__("Charge")
        self._encode_as_onehot = encode_as_onehot
        self._min_known_charge = 0
        self._max_known_charge = 0

    def prepare_metadata(self, atom: Atom) -> None:
        self._assert_metadata_uninitialised()
        self._min_known_charge = min(self._min_known_charge, atom.GetFormalCharge())
        self._max_known_charge = max(self._max_known_charge, atom.GetFormalCharge())

    @property
    def feature_type(self) -> FeatureType:
        if self._encode_as_onehot:
            return FeatureType.CategoryId
        else:
            return FeatureType.PositiveNumber

    @property
    def feature_width(self) -> int:
        if self._encode_as_onehot:
            # Note that we use the first entry of the one-hot vector to signify "unknown"
            return 1 + self._max_known_charge - self._min_known_charge + 1
        else:
            return 1

    def featurise(self, atom: Atom) -> np.ndarray:
        self._assert_metadata_initialised()
        if self._encode_as_onehot:
            features = np.zeros(shape=(self.feature_width,))
            features[1 + atom.GetFormalCharge() - self._min_known_charge] = 1.0
            return features
        else:
            return np.array([atom.GetFormalCharge()])


class AtomNumRadicalEletronsFeatureExtractor(AtomFeatureExtractor):
    def __init__(self, encode_as_onehot: bool = False):
        super().__init__("NumRadicalElectrons")
        self._encode_as_onehot = encode_as_onehot
        self._min_known_num = 0
        self._max_known_num = 0

    def prepare_metadata(self, atom: Atom) -> None:
        self._assert_metadata_uninitialised()
        self._min_known_num = min(self._min_known_num, atom.GetNumRadicalElectrons())
        self._max_known_num = max(self._max_known_num, atom.GetNumRadicalElectrons())

    @property
    def feature_type(self) -> FeatureType:
        if self._encode_as_onehot:
            return FeatureType.CategoryId
        else:
            return FeatureType.PositiveNumber

    @property
    def feature_width(self) -> int:
        if self._encode_as_onehot:
            # Note that we use the first entry of the one-hot vector to signify "unknown"
            return 1 + self._max_known_num - self._min_known_num + 1
        else:
            return 1

    def featurise(self, atom: Atom) -> np.ndarray:
        self._assert_metadata_initialised()
        if self._encode_as_onehot:
            features = np.zeros(shape=(self.feature_width,))
            features[1 + atom.GetNumRadicalElectrons() - self._min_known_num] = 1.0
            return features
        else:
            return np.array([atom.GetNumRadicalElectrons()])


class AtomIsotopeFeatureExtractor(AtomFeatureExtractor):
    def __init__(self):
        super().__init__("Isotope")
        self._metadata_initialised = True

    @property
    def feature_type(self) -> FeatureType:
        return FeatureType.PositiveNumber

    @property
    def feature_width(self) -> int:
        return 1

    def featurise(self, atom: Atom) -> np.ndarray:
        return np.array([atom.GetIsotope()])


class AtomMassFeatureExtractor(AtomFeatureExtractor):
    def __init__(self):
        super().__init__("Mass")
        self._metadata_initialised = True

    @property
    def feature_type(self) -> FeatureType:
        return FeatureType.PositiveNumber

    @property
    def feature_width(self) -> int:
        return 1

    def featurise(self, atom: Atom) -> np.ndarray:
        return np.array([atom.GetMass()])


class AtomValenceFeatureExtractor(AtomFeatureExtractor):
    def __init__(self):
        super().__init__("Valence")
        self._metadata_initialised = True

    @property
    def feature_type(self) -> FeatureType:
        return FeatureType.PositiveNumber

    @property
    def feature_width(self) -> int:
        return 1

    def featurise(self, atom: Atom) -> np.ndarray:
        return np.array([atom.GetTotalValence()])


class AtomNumHydrogensFeatureExtractor(AtomFeatureExtractor):
    def __init__(self):
        super().__init__("NumHydrogens")
        self._metadata_initialised = True

    @property
    def feature_type(self) -> FeatureType:
        return FeatureType.PositiveNumber

    @property
    def feature_width(self) -> int:
        return 1

    def featurise(self, atom: Atom) -> np.ndarray:
        return np.array([atom.GetTotalNumHs()])


class AtomIsAromaticFeatureExtractor(AtomFeatureExtractor):
    def __init__(self):
        super().__init__("IsAromatic")
        self._metadata_initialised = True

    @property
    def feature_type(self) -> FeatureType:
        return FeatureType.BoolValue

    @property
    def feature_width(self) -> int:
        return 1

    def featurise(self, atom: Atom) -> np.ndarray:
        features = np.zeros(shape=(self.feature_width,))
        features[0] = float(atom.GetIsAromatic())
        return features


class AtomRingInformationExtractor(AtomFeatureExtractor):
    MIN_RING_SIZE_TO_CONSIDER = 3
    MAX_RING_SIZE_TO_CONSIDER = 15

    def __init__(self, encode_ring_sizes: bool = True):
        super().__init__("RingInformation")
        self._encode_ring_sizes = encode_ring_sizes

    @property
    def feature_type(self) -> FeatureType:
        if self._encode_ring_sizes:
            return FeatureType.Other
        else:
            return FeatureType.BoolValue

    @property
    def feature_width(self) -> int:
        if self._encode_ring_sizes:
            # Note that we use the first entry to signify "in ring of any size"
            return 1 + self.MAX_RING_SIZE_TO_CONSIDER - self.MIN_RING_SIZE_TO_CONSIDER
        else:
            return 1

    def featurise(self, atom: Atom) -> np.ndarray:
        self._assert_metadata_initialised()
        features = np.zeros(shape=(self.feature_width,))
        features[0] = float(atom.IsInRing())

        if self._encode_ring_sizes:
            for ring_size in range(self.MIN_RING_SIZE_TO_CONSIDER, self.MAX_RING_SIZE_TO_CONSIDER):
                if atom.IsInRingSize(ring_size):
                    features[1 + ring_size - self.MIN_RING_SIZE_TO_CONSIDER] = 1.0

        return features


def get_default_atom_featurisers():
    """Creates and returns a list of fresh atom featurisers."""
    return [
        AtomTypeFeatureExtractor(),
        AtomDegreeFeatureExtractor(),
        AtomChargeFeatureExtractor(),
        AtomNumRadicalEletronsFeatureExtractor(),
        AtomIsotopeFeatureExtractor(),
        AtomMassFeatureExtractor(),
        AtomValenceFeatureExtractor(),
        AtomNumHydrogensFeatureExtractor(),
        AtomIsAromaticFeatureExtractor(),
        AtomRingInformationExtractor(),
    ]
