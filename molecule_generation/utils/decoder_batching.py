from typing import Dict, Any, Callable, List, Generator, Tuple, Optional
import logging


import numpy as np

from molecule_generation.utils.moler_decoding_utils import MoLeRDecoderState
from molecule_generation.chem.molecule_dataset_utils import BOND_DICT
from molecule_generation.chem.atom_feature_utils import AtomTypeFeatureExtractor
from molecule_generation.chem.motif_utils import MotifVocabulary

logger = logging.getLogger(__name__)


def batch_decoder_states(
    *,
    max_nodes_per_batch: int,
    atom_featurisers: List[AtomTypeFeatureExtractor],
    motif_vocabulary: Optional[MotifVocabulary],
    uses_categorical_features: bool,
    decoder_states: List[MoLeRDecoderState],
    init_batch_callback: Callable[[Dict[str, Any]], None],
    add_state_to_batch_callback: Callable[[Dict[str, Any], MoLeRDecoderState], None],
) -> Generator[Tuple[Dict[str, Any], List[MoLeRDecoderState]], None, None]:
    graph_id_offset = 0

    def _get_empty_batch() -> Dict[str, Any]:
        new_batch = {
            "nodes_in_batch": 0,
            "graphs_in_batch": 0,
            "molecule_representations": [],
            "node_features": [],
            "node_categorical_features": [],
            "node_to_graph_map": [],  # Local to the batch, starts at 0 in each yielded batch.
            "adjacency_lists": [[] for _ in range(len(BOND_DICT))],
        }

        init_batch_callback(new_batch)
        return new_batch

    def _flush(old_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Yield a batch and start a new one"""
        nonlocal graph_id_offset
        _finalise_batch(old_batch, uses_categorical_features)
        yield (
            old_batch,
            decoder_states[graph_id_offset : graph_id_offset + old_batch["graphs_in_batch"]],
        )
        graph_id_offset += old_batch["graphs_in_batch"]
        return _get_empty_batch()

    batch = _get_empty_batch()

    for decoder_state in decoder_states:
        node_features, node_categorical_features = decoder_state.get_node_features(
            atom_featurisers, motif_vocabulary
        )
        mol_num_nodes = node_features.shape[0]

        if mol_num_nodes >= max_nodes_per_batch:
            logger.warning(
                f"Single graph of {mol_num_nodes} nodes exceeds max_nodes_per_batch {max_nodes_per_batch}"
            )
            if batch["graphs_in_batch"] > 0:
                batch = yield from _flush(batch)

        batch["molecule_representations"].append(decoder_state.molecule_representation)
        batch["node_features"].append(node_features)
        batch["node_to_graph_map"].extend([batch["graphs_in_batch"]] * mol_num_nodes)

        if uses_categorical_features:
            batch["node_categorical_features"].append(node_categorical_features)

        for edge_type_idx, mol_adj_list in enumerate(decoder_state.adjacency_lists):
            if len(mol_adj_list) > 0:
                batch["adjacency_lists"][edge_type_idx].append(
                    np.array(mol_adj_list, dtype=np.int32) + batch["nodes_in_batch"]
                )

        add_state_to_batch_callback(batch, decoder_state)

        batch["nodes_in_batch"] += mol_num_nodes
        batch["graphs_in_batch"] += 1
        if batch["nodes_in_batch"] >= max_nodes_per_batch:
            batch = yield from _flush(batch)

    if batch["graphs_in_batch"] > 0:
        # Yield the last few samples
        yield from _flush(batch)


def _finalise_batch(batch: Dict[str, Any], uses_categorical_features: bool) -> None:
    """Finalise batch contents"""
    batch["node_features"] = np.concatenate(batch["node_features"], axis=0)

    if uses_categorical_features:
        batch["node_categorical_features"] = np.concatenate(
            batch["node_categorical_features"], axis=0
        )

    for edge_type_idx, type_adj_lists in enumerate(batch["adjacency_lists"]):
        num_edges = sum(len(adj_list) for adj_list in type_adj_lists)
        if num_edges == 0:
            batch["adjacency_lists"][edge_type_idx] = np.zeros((0, 2), dtype=np.int32)
        else:
            batch["adjacency_lists"][edge_type_idx] = np.concatenate(type_adj_lists, axis=0)

    # Insert self-loops (cf. TraceDataset._finalise_batch)
    batch["adjacency_lists"].append(
        np.repeat(np.arange(batch["nodes_in_batch"], dtype=np.int32), 2).reshape(-1, 2)
    )
    batch["adjacency_lists"] = tuple(batch["adjacency_lists"])
