"""Function to constrain edges in molecular dataset."""
from typing import Collection, List

import numpy as np

# Copied from the RDKit definitions, found in https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/atomic_data.cpp
from molecule_generation.chem.rdkit_helpers import get_true_symbol, get_charge_from_symbol

ATOM_TO_MAX_VALENCE = {
    "C": 4,
    "Br": 1,
    "N": 3,
    "O": 2,
    "S": 6,
    "Cl": 1,
    "F": 1,
    "P": 7,
    "I": 5,
    "B": 3,
    "Si": 6,
    "Se": 6,
}


def constrain_edge_choices_based_on_valence(
    start_node: int,
    candidate_target_nodes: np.ndarray,
    adjacency_lists: Collection[np.ndarray],
    node_types: List[str],
) -> np.ndarray:
    """Function to constrain the valid edge choices based on the given adjacency lists, and
    maximum valency of nodes.

    Args:
        start_node: Node from which edges are to be drawn.
        candidate_target_nodes: Nodes to which edges are to be drawn.
        adjacency_lists: a list of numpy arrays corresponding to the adjacency lists of the
            graph to which one of the edges is to be added. We assume that there
            are 3 edge types: single, double and triple bonds. There are stored in the 0th, 1st and
            2nd array in the adjacency_lists object, respectively. We further assume that the
            adjacency lists are symmetric, so if (u, v) is in one of the lists, then (v, u) is in
            the same list.
        node_types: a list of node type information, e.g. ["C", "N++", ...].
            Shape = [num_nodes]

    Returns:
        A bool mask of the same length as candidate_target_nodes (as np array), indicating which
        of these choices are valid.
    """
    assert len(adjacency_lists) == 3, "We can only deal with 3 edge types for now."

    # Early exit for degenerate case:
    if len(candidate_target_nodes) == 0:
        return np.zeros(shape=(0,), dtype=bool)

    node_idx_to_valency_map = _calculate_valency_map(adjacency_lists, node_types)
    node_idx_to_max_valency_map = _calculate_max_valency(node_types)
    node_idx_to_open_map = node_idx_to_valency_map < node_idx_to_max_valency_map
    edge_to_open_node = node_idx_to_open_map[candidate_target_nodes]
    focus_node_is_open = node_idx_to_open_map[start_node]
    return np.logical_and(edge_to_open_node, focus_node_is_open)


def constrain_edge_types_based_on_valence(
    start_node: int,
    candidate_target_nodes: np.ndarray,
    adjacency_lists: List[np.ndarray],
    node_types: List[np.ndarray],
) -> np.ndarray:
    """Calculate the valid edge type mask based on the given edge choices and graph.

    Args:
        start_node: Node from which edges are to be drawn.
        candidate_target_nodes: Nodes to which edges are to be drawn.
        adjacency_lists: a list of numpy arrays corresponding to the adjacency lists of the
            graph to which one of the edges from `edge_choices` is to be added. We assume that there
            are 3 edge types: single, double and triple bonds. These are stored in the 0th, 1st and
            2nd array in the adjacency_lists object, respectively. We further assume that the
            adjacency lists are symmetric, so if (u, v) is in one of the lists, then (v, u) is in
            the same list.
        node_types: a list of node type information. Shape = [num_nodes]

    Returns:
        A numpy array of shape (E, num_edge_types) which represents a mask for the valid edge types
        for each edge. As an example, element `(e, t) == 1` if edge `e` of `edge_choices` can be of
        type `t`, and 0 otherwise.
    """
    num_edge_types = len(adjacency_lists)

    # Early out for degenerate case:
    if len(candidate_target_nodes) == 0:
        return np.zeros(shape=(0, 3), dtype=np.float64)

    node_idx_to_valency_map = _calculate_valency_map(adjacency_lists, node_types)
    node_idx_to_max_valency_map = _calculate_max_valency(node_types)
    node_idx_to_open_slots = node_idx_to_max_valency_map - node_idx_to_valency_map
    focus_node_open_slots = node_idx_to_open_slots[start_node]
    node_idx_to_open_slots = np.minimum(node_idx_to_open_slots, focus_node_open_slots)
    node_idx_to_valid_incoming_edge_mask = _calculate_valid_edge_mask(
        node_idx_to_open_slots, num_edge_types
    )
    return node_idx_to_valid_incoming_edge_mask[candidate_target_nodes]


def _calculate_valid_edge_mask(
    node_idx_to_open_slots: np.ndarray, num_edge_types: int
) -> np.ndarray:
    """Calculate the valid edge type mask based on the 'open slots' in the set of nodes.

    Args:
        node_idx_to_open_slots: an array whose integer values are the number of edges that a node
            could have attached to it.
        num_edge_types: the number of edge types.

    Returns:
        A mask

    >>> node_idx_to_open_slots = np.array([1, 2, 1, 3, 5])
    >>> _calculate_valid_edge_mask(node_idx_to_open_slots, 3)
    array([[1., 0., 0.],
           [1., 1., 0.],
           [1., 0., 0.],
           [1., 1., 1.],
           [1., 1., 1.]])
    """
    max_valence_for_node_types = num_edge_types
    node_idx_to_max_valence_to_remove = np.minimum(
        node_idx_to_open_slots, max_valence_for_node_types
    )

    edge_type_mask = 1 - np.triu(
        np.ones(shape=(num_edge_types + 1, num_edge_types), dtype=np.float64)
    )
    return edge_type_mask[node_idx_to_max_valence_to_remove]


def _calculate_valency_map(
    adjacency_lists: Collection[np.ndarray],
    node_types: List[str],
) -> np.ndarray:
    """Calculate a numpy array of the valencies of each node.

    Args:
        adjacency_lists: a list of numpy arrays corresponding to the adjacency lists of the graph.
            The assumptions we make about this parameter are the same as for the `constrain_*`
            functions above.
        node_types: a list of node type information. Shape = [num_nodes]

    Returns:
        A numpy array of shape (num_nodes,). The ith element of the array corresponds to the valency
        of the node whose index is i.
    """
    node_idx_to_valency_map = np.zeros(shape=len(node_types), dtype=np.int32)
    for edge_type_idx, adjacency_list in enumerate(adjacency_lists):
        if len(adjacency_list) == 0:
            continue
        node_idx, count = np.unique(adjacency_list[:, 0], return_counts=True)
        count *= edge_type_idx + 1  # adjust for valency of edge type.
        node_idx_to_valency_map[node_idx] += count
    return node_idx_to_valency_map


def _calculate_max_valency(node_types: List[str]) -> np.ndarray:
    """Calculate the maximum valency for each of the nodes in the input array.

    Args:
        node_types: a list of node type information. Shape = [num_nodes]

    Returns:
        A numpy array of shape (num_nodes,). The ith element of the array corresponds to the maximum
        valency of the node whose index is i, based on its type.
    """
    node_idx_to_max_valency_map = np.array(
        [_max_valence_for_single_node(node_type) for node_type in node_types]
    )
    return node_idx_to_max_valency_map


def _max_valence_for_single_node(node_type: str) -> int:
    node_symbol = get_true_symbol(node_type)
    charge = get_charge_from_symbol(node_type)
    return ATOM_TO_MAX_VALENCE[node_symbol] + charge


if __name__ == "__main__":
    import doctest

    doctest.testmod()
