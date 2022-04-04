#!/usr/bin/env python3
from multiprocessing import Pool
from typing import Dict, Any, Tuple, Optional, List, Set, Iterator, Iterable, Type

import numpy as np
from dpu_utils.utils import RichPath
from rdkit import Chem
from tf2_gnn import DataFold
from tf2_gnn.data.graph_dataset import GraphSampleType

from molecule_generation.dataset.trace_dataset import TraceDataset, TraceSample
from molecule_generation.chem.motif_utils import (
    MotifVocabulary,
    find_motifs_from_vocabulary,
    get_motif_type_to_node_type_index_map,
)
from molecule_generation.chem.molecule_dataset_utils import (
    _smiles_to_rdkit_mol,
    molecule_to_graph,
    AtomFeatureExtractor,
)
from molecule_generation.preprocessing.data_conversion_utils import (
    _convert_single_jsonl_to_graph_sample,
    _convert_graph_sample_to_graph_trace,
)
from molecule_generation.preprocessing.generation_order import BFSOrder, GenerationOrder


def transform_smiles_to_samples(
    feature_extractors: List[AtomFeatureExtractor],
    property_names: Iterable[str],
    smiles: str,
    include_generation_trace: bool = True,
    MoLeR_style_trace: bool = False,
    generation_order_cls: Type[GenerationOrder] = BFSOrder,
    motif_vocabulary: Optional[MotifVocabulary] = None,
):
    """Convert a SMILES string to a molecule and a trace sample.

    Note: Returns None if conversion fails.
    """
    # Transform data:
    mol_info_dict = _smiles_to_rdkit_mol({"SMILES": smiles})

    if motif_vocabulary is None:
        mol_info_dict["motifs"] = []
    else:
        mol_info_dict["motifs"] = find_motifs_from_vocabulary(
            molecule=mol_info_dict["mol"], motif_vocabulary=motif_vocabulary
        )

    try:
        mol_info_dict["graph"] = molecule_to_graph(
            mol=mol_info_dict["mol"],
            atom_feature_extractors=feature_extractors,
            motif_vocabulary=motif_vocabulary,
            motifs=mol_info_dict["motifs"],
        )
    except ValueError as e:
        print(
            f"Warning! Failed to convert SMILES string to graph. "
            f"Given SMILES string:\n{smiles}\n"
            f"Error received: {e}"
        )
        return None

    # Insert dummy values for all known properties, to trigger computation:
    for prop_name in property_names:
        if prop_name not in mol_info_dict["properties"]:
            mol_info_dict["properties"][prop_name] = -1
    graph_sample = _convert_single_jsonl_to_graph_sample(mol_info_dict)

    trace_sample = _convert_graph_sample_to_graph_trace(
        graph=graph_sample,
        num_fwd_edge_types=graph_sample.num_edge_types,
        tie_fwd_bkwd_edges=True,
        add_self_loop_edges=False,  # Added on the fly in the dataset batching, if needed.
        atom_feature_extractors=feature_extractors,
        motif_vocabulary=motif_vocabulary,
        include_generation_trace=include_generation_trace,
        MoLeR_style_trace=MoLeR_style_trace,
        generation_order_cls=generation_order_cls,
        save_molecule_to_trace=MoLeR_style_trace,
    )

    # Check if an exception occurred, and re-raise if necessary
    if isinstance(trace_sample, tuple):
        print(
            f"Warning! Failed to convert graph to trace. "
            f"Given SMILES string:\n{smiles}\n"
            f"Error received: {trace_sample[1]}"
        )
        return None

    return trace_sample.mol, trace_sample


class InMemoryTraceDataset(TraceDataset):
    def __init__(
        self,
        params: Dict[str, Any],
        metadata: Dict[str, Any],
        num_edge_types: int,
        node_feature_shape: Tuple[int, ...],
        node_type_index_to_string: Dict[int, str],
        partial_node_feature_shape: Tuple[int, ...] = None,
        MoLeR_style_traces: bool = False,
    ):
        super().__init__(params, metadata)

        self._num_edge_types = num_edge_types
        self._node_type_index_to_string = node_type_index_to_string.copy()

        self._motif_vocabulary = self.metadata.get("motif_vocabulary")
        self._generation_order_cls = self.metadata.get("generation_order", BFSOrder)

        node_categorical_num_classes = self.metadata.get("_node_categorical_num_classes")

        if self._motif_vocabulary is not None:
            self._motif_to_node_type_index = get_motif_type_to_node_type_index_map(
                motif_vocabulary=self._motif_vocabulary,
                num_atom_types=len(self._node_type_index_to_string),
            )

            for motif, node_type in self._motif_to_node_type_index.items():
                self._node_type_index_to_string[node_type] = motif

            if node_categorical_num_classes is None:
                # Fill in the number of categorical classes and fix feature shape for older models.
                node_categorical_num_classes = len(self._node_type_index_to_string)
                node_feature_shape = (node_feature_shape[0] - node_categorical_num_classes,)
        else:
            self._motif_to_node_type_index = {}

        # This will almost always be the case.
        if partial_node_feature_shape is None:
            partial_node_feature_shape = node_feature_shape

        self._node_feature_shape = node_feature_shape
        self._partial_node_feature_shape = partial_node_feature_shape
        self._node_categorical_num_classes = node_categorical_num_classes

        self._node_type_to_index_map = {
            typ: idx for idx, typ in self._node_type_index_to_string.items()
        }

        self._MoLeR_style_traces = MoLeR_style_traces

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def params(self) -> Dict[str, Any]:
        return self._params

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @property
    def num_edge_types(self) -> int:
        return self._num_edge_types

    @property
    def node_feature_shape(self) -> Tuple:
        return self._node_feature_shape

    @property
    def partial_node_feature_shape(self) -> Tuple:
        return self._partial_node_feature_shape

    @property
    def node_categorical_num_classes(self) -> Optional[int]:
        return self._node_categorical_num_classes

    @property
    def node_type_index_to_string(self) -> Dict[int, str]:
        return self._node_type_index_to_string

    @property
    def motif_vocabulary(self) -> MotifVocabulary:
        return self._motif_vocabulary

    def node_type_to_index(self, node_type: str) -> int:
        if node_type in self._motif_to_node_type_index:
            return self._motif_to_node_type_index[node_type]
        else:
            return self._node_type_to_index_map.get(node_type, 0)

    def node_types_to_multi_hot(self, node_types: List[str]) -> np.ndarray:
        if not self._MoLeR_style_traces:
            return super().node_types_to_multi_hot(node_types)
        correct_indices = self.node_types_to_indices(node_types)
        multihot = np.zeros(shape=(self.num_node_types,), dtype=np.float32)
        for idx in correct_indices:
            multihot[idx] = 1.0
        return multihot

    def load_data(self, path: RichPath, folds_to_load: Optional[Set[DataFold]] = None) -> None:
        raise ValueError("Attempting to load data from a path into DummyGraphDataset!")

    def load_data_from_list(
        self, datapoints: List[TraceSample], target_fold: DataFold = DataFold.TEST
    ):
        if target_fold not in self._loaded_data:
            self._loaded_data[target_fold] = []
        for datapoint in datapoints:
            self._loaded_data[target_fold].append(datapoint)

    def _graph_iterator(self, data_fold: DataFold) -> Iterator[GraphSampleType]:
        if data_fold == DataFold.TRAIN:
            np.random.shuffle(self._loaded_data[data_fold])
        return iter(self._loaded_data[data_fold])

    def transform_smiles_to_sample(
        self, smiles: str, include_generation_trace: bool = True
    ) -> Optional[Tuple[Chem.Mol, TraceSample]]:
        """Convert a SMILES string to a molecule and a trace sample.

        Note: Returns None if conversion fails.
        """
        return transform_smiles_to_samples(
            self.metadata["feature_extractors"],
            self._graph_property_names,
            smiles,
            include_generation_trace,
            self._MoLeR_style_traces,
            self._generation_order_cls,
            self.motif_vocabulary,
        )

    def transform_many_smiles_to_samples(
        self, smiles_list: List[str], include_generation_trace: bool = True
    ) -> List[Optional[Tuple[Chem.Mol, TraceSample]]]:
        """Convert a list of SMILES strings to a list of  molecule and a trace sample tuples.

        Note: Returns None for every SMILES string in which conversion fails.
        """
        with Pool() as p:
            return p.starmap(
                func=transform_smiles_to_samples,
                iterable=(
                    (
                        self.metadata["feature_extractors"],
                        self._graph_property_names,
                        smiles,
                        include_generation_trace,
                        self._MoLeR_style_traces,
                        self._generation_order_cls,
                        self.motif_vocabulary,
                    )
                    for smiles in smiles_list
                ),
            )
