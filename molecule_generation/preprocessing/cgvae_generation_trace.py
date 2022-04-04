"""Function (and helpers) to map a GraphSample to a GraphTraceSample."""
import random
from collections import defaultdict
from copy import deepcopy
from enum import Enum
from queue import Queue
from typing import Dict, List, Union, Set, Tuple

import numpy as np
from rdkit.Chem import BondType, RWMol

from molecule_generation.preprocessing.graph_sample import (
    AdjacencyList,
    Edge,
    GraphSample,
    GraphTraceSample,
)
from molecule_generation.chem.rdkit_helpers import initialise_atom_from_symbol


class NodeState(Enum):
    """The representation of the node state in the exploration process.

    Every node will move through these states in order.
    """

    UNDISCOVERED = 0
    DISCOVERED = 1
    FOCUS = 2
    LOCKED = 3


EdgeCollection = List[Edge]

EDGE_TYPE_IDX_TO_RDKIT_TYPE = {
    -1: BondType.SINGLE,  # This is used in the code where the actual bond type does not matter, but we need to pick one.
    0: BondType.SINGLE,
    1: BondType.DOUBLE,
    2: BondType.TRIPLE,
}


def graph_sample_to_cgvae_trace(graph_sample: GraphSample) -> GraphTraceSample:
    """Randomly generate a GraphTraceSample from a single GraphSample.

    Note:
    We assume, for now, that the GraphSample contains only undirected edges. Practically, we take
    that assumption to mean that for each edge (u, v), there is _exactly one_ representative in the
    GraphSample adjacency lists. So, if (u, v) is in the adjacency lists, then (v, u) is not.

    Example use:
    >>> adjacency_list = [Edge(source=0, target=1, type=0),
    ...                   Edge(source=0, target=2, type=0),
    ...                   Edge(source=1, target=2, type=1)]
    >>> graph_sample = GraphSample(adjacency_list=adjacency_list,
    ...                            num_edge_types=4,
    ...                            node_features=[0., 0., 0.],
    ...                            graph_properties={"sa_score": 8.9},
    ...                            node_types=["C", "F", "C"],
    ...                            smiles_string="not really a SMILES string",
    ...                            )
    ...
    >>> graph_trace = graph_sample_to_cgvae_trace(graph_sample)
    >>> assert len(graph_trace.partial_graphs) == 6
    >>> assert len(graph_trace.correct_edge_choices) == 6
    >>> assert len(graph_trace.valid_edge_choices) == 6
    >>> assert len(graph_trace.focus_nodes) == 6

    Args:
        graph_sample: a GraphSample object from which we want to sample a generation trace.

    Returns:
        A GraphTraceSample representing a single generation trace, sampled at random from all of the
            possible ways of generating the given graph_sample.

    """
    num_nodes = len(graph_sample.node_features)
    # Initialise the node states
    node_states = {node_idx: NodeState.UNDISCOVERED for node_idx in range(num_nodes)}

    # Initialise the empty RWMol:
    mol = RWMol()
    for node_symbol in graph_sample.node_types:
        mol.AddAtom(initialise_atom_from_symbol(node_symbol))

    # First, pick a node at random:
    starting_node_idx = random.choice(range(num_nodes))
    exploration_queue = Queue(maxsize=num_nodes)
    exploration_queue.put(starting_node_idx)

    # This is where the undirected assumption on the `graph_sample` is used.
    adjacency_list = __symmetrize_adjacency_list(graph_sample.adjacency_list)

    # Calculate a dictionary of node_idx -> list of neighboring nodes.
    node_idx_to_neighbours = calculate_source_to_targets_dict(adjacency_list)

    # We will build the graph_sample adjacency_list one edge at a time, recording our progress in
    # a this (growing) partial_adjacency_list:
    partial_adjacency_list: AdjacencyList = []

    # We keep the history of the partial adjacency lists in this generation_trace list:
    generation_trace: List[AdjacencyList] = []

    # At each step in the graph generation, we keep the edges that we _could_ have chosen.
    # correct_edge_choices[i] will be the collection of all edges we could have added to
    # the adjacency list in generation_trace[i] to get a correct partial adjacency list in
    # generation_trace[i+1]
    correct_edge_choices: List[EdgeCollection] = []

    # At each step, we also record edges to nodes which are not locked. These correspond to edges
    # that are "valid" in terms of our exploration strategy, but which do not necessarily correspond
    # to edges in the complete graph. They are valid choices, but not always correct choices.
    valid_edge_choices: List[EdgeCollection] = []

    # We need to keep track of the (graph) distance between the focus node and all of the target
    # nodes in the valid edge choices.
    graph_distance_to_target: List[List[int]] = []

    # Here, we record the order in which nodes are brought into focus.
    focus_nodes: List[int] = []

    # And here we record the generation trace in the format of an RDKit molecule.
    molecule_trace: List[RWMol] = []

    while True:
        if exploration_queue.empty():
            break
        focus_node_idx = exploration_queue.get()
        node_states[focus_node_idx] = NodeState.FOCUS
        edges_to_add = get_node_edges(node_idx_to_neighbours, focus_node_idx, node_states)
        edges_to_open_nodes = get_open_edges(focus_node_idx, node_states)
        random.shuffle(edges_to_add)
        while edges_to_add:
            focus_nodes.append(focus_node_idx)

            # Record all possible edges we could add at this step:
            correct_edge_choices.append(deepcopy(edges_to_add))

            # Record all valid open edges for this step:
            valid_edge_choices.append(deepcopy(edges_to_open_nodes))

            graph_distance_to_target.append(
                calculate_dist_from_focus_to_valid_target(
                    adjacency_list=partial_adjacency_list,
                    focus_node=focus_node_idx,
                    target_nodes=[edge.target for edge in edges_to_open_nodes],
                )
            )
            molecule_trace.append(RWMol(mol))

            # Get the edge that we actually do add:
            edge = edges_to_add.pop()
            # Add it to the molecule:
            mol.AddBond(edge.source, edge.target, EDGE_TYPE_IDX_TO_RDKIT_TYPE[edge.type])
            # And remove that edge from the 'edges_to_open_nodes' list.
            edges_to_open_nodes.remove(Edge(source=edge.source, target=edge.target, type=-1))

            # Record the current state of the partial adjacency list, then add the selected edge:
            generation_trace.append(deepcopy(partial_adjacency_list))
            partial_adjacency_list.append(edge)

            # Add the non-focus node to the exploration queue if this is the first time we have
            # seen it.
            target_node_idx = node_idx_if_undiscovered(edge, node_states)
            if target_node_idx is not None:
                exploration_queue.put(target_node_idx)

        # Add a step where there are no correct edge choices, so that we can learn what these look
        # like as well.
        focus_nodes.append(focus_node_idx)
        correct_edge_choices.append([])
        valid_edge_choices.append(deepcopy(edges_to_open_nodes))
        graph_distance_to_target.append(
            calculate_dist_from_focus_to_valid_target(
                adjacency_list=partial_adjacency_list,
                focus_node=focus_node_idx,
                target_nodes=[edge.target for edge in edges_to_open_nodes],
            )
        )
        molecule_trace.append(RWMol(mol))
        generation_trace.append(deepcopy(partial_adjacency_list))
        node_states[focus_node_idx] = NodeState.LOCKED

    result = convert_lists_to_graph_trace_sample(
        graph_sample,
        generation_trace,
        correct_edge_choices,
        valid_edge_choices,
        focus_nodes,
        graph_distance_to_target,
        molecule_trace,
    )
    return result


def _convert_np_to_edgecollection(arrays: Union[List[np.ndarray], np.ndarray]) -> EdgeCollection:
    """Convert a numpy array or list of arrays, representing adjacency lists, into an EdgeCollection."""
    if isinstance(arrays, np.ndarray):
        arrays = [arrays]

    edge_collection: EdgeCollection = []
    for i, array in enumerate(arrays):
        for edge in array:
            edge_collection.append(Edge(source=edge[0], target=edge[1], type=i))
    return edge_collection


def calculate_dist_from_focus_to_valid_target(
    adjacency_list: Union[
        EdgeCollection, np.ndarray, List[np.ndarray], List[List[Tuple[int, int]]]
    ],
    focus_node: int,
    target_nodes: List[int],
    symmetrise_adjacency_list: bool = True,
) -> List[int]:
    """Calculate the graph distance between the focus node and the targets of the valid edges.

    Args:
        adjacency_list: the adjacency list representation of the graph. We allow four formats:
            * EdgeCollection (i.e., an iterable over Edge NamedTuples)
            * Tensor of Edges, i.e., int32 tensor of shape [None, 2]
            * List of tensors of edges (one per edge type, each of shape [None, 2])
            * List of lists of pairs.
        focus_node: the index of the node to which we want the distances calculated.
        target_nodes: the nodes in whose distances from focus_node we are interested.
        symmetrise_adjacency_list: bool representing whether the adjacency list provided needs to be symmetrised, or
            already is. The main body of this function requires that the adjacency list is symmetric: if it contains the
            edge (u, v), it must also contain (v, u). If that is already the case, this parameter should be false. If
            is not the case, and the adjacency list contains only one representative of (u, v) and (v, u), then this
            parameter should be true.

    Returns:
        The graph distance between the focus node and the target of each of the valid edges. The
        order between these distances and the valid edges is preserved! So, if the edge
        valid_edges[i] has target node v, then the distance between the focus node and v is in index
        i of the returned list.

    Note:
        Unconnected nodes will have a distance of `0`

    """
    # Easy out for the degenerate cases:
    if len(target_nodes) == 0:
        return []

    if len(adjacency_list) == 0:
        return [0] * len(target_nodes)

    # Build node-to-neighbour map, ignoring edge types:
    node_idx_to_neighbours: Dict[int, Set[int]] = defaultdict(set)
    if isinstance(adjacency_list, np.ndarray) and len(adjacency_list.shape) == 2:
        adjacency_lists = [adjacency_list]
    elif (
        isinstance(adjacency_list, list)
        and len(adjacency_list) > 0
        and isinstance(adjacency_list[0], Edge)
    ):
        adjacency_lists = [adjacency_list]
    else:
        adjacency_lists = adjacency_list
    for adj_list in adjacency_lists:
        for edge in adj_list:
            if isinstance(edge, Edge):
                source_idx, target_idx = edge.source, edge.target
            else:
                source_idx, target_idx = edge[0], edge[1]
            node_idx_to_neighbours[source_idx].add(target_idx)
            if symmetrise_adjacency_list:
                node_idx_to_neighbours[target_idx].add(source_idx)

    # Breadth first search of the graph defined by the given adjacency_list
    exploration_queue: Queue[int] = Queue()
    exploration_queue.put(focus_node)
    distance_dict = {focus_node: 0}

    # Stop BFS once we've discovered all targets we are interested in:
    valid_target_set = set(target_nodes)
    while len(valid_target_set) > 0:
        if exploration_queue.empty():
            break
        current_node = exploration_queue.get()
        for neighbour in node_idx_to_neighbours[current_node]:
            if neighbour not in distance_dict:
                exploration_queue.put(neighbour)
                distance_dict[neighbour] = distance_dict[current_node] + 1
                valid_target_set.discard(neighbour)

    # Note: Unconnected nodes will have a distance of `0`.
    distance_to_targets = [distance_dict.get(target, 0) for target in target_nodes]
    return distance_to_targets


def convert_lists_to_graph_trace_sample(
    graph_sample: GraphSample,
    generation_trace: List[AdjacencyList],
    correct_edge_choices: List[EdgeCollection],
    valid_edge_choices: List[EdgeCollection],
    focus_nodes: List[int],
    graph_distance_to_target: List[List[int]],
    molecule_trace: List[RWMol],
) -> GraphTraceSample:
    """Convert various generation trace related lists to a single GraphTraceSample object.

    Args:
        graph_sample: the original GraphSample object that the generation trace is made from.
        generation_trace: the sequence of adjacency lists that make up the generation trace.
        correct_edge_choices: the collection of all correct edges at each stage in the generation
            trace. For example, correct_edge_choices[i] should contain all the edges in graph_sample
            that are connected to focus_nodes[i] which are _not_ already in generation_trace[i].
        valid_edge_choices: the collection of all possible edge choices at each stage in the
            generation trace. A possible edge choice is defined as an edge starting at the focus
            node and ending at a node which is not locked (or the focus node).
        focus_nodes: a list containing the index of the focus node at each step in the generation
            trace.
        graph_distance_to_target: a list of lists, each containing the graph distance between the
            focus node and the valid edge choice targets at each step of the generation trace.
        molecule_trace: a list of RWMol objects which mirror the generation trace of the partial graphs.

    Returns:
        A GraphTraceSample object containing all of the information of this particular graph
        generation.

    """
    # Make sure we have not made an error in the generation/molecule trac:
    assert len(generation_trace) == len(
        molecule_trace
    ), "Generation trace and molecule trace are different lengths."
    node_features = graph_sample.node_features
    graph_properties = graph_sample.graph_properties
    smiles_string = graph_sample.smiles_string
    num_edge_types = graph_sample.num_edge_types
    node_types = graph_sample.node_types
    partial_graphs = [
        GraphSample(
            adjacency_list=adjacency_list,
            num_edge_types=num_edge_types,
            node_features=node_features,
            graph_properties=graph_properties,
            node_types=node_types,
            smiles_string=smiles_string,
            mol=mol,
        )
        for adjacency_list, mol in zip(generation_trace, molecule_trace)
    ]

    # Double check that we have added the same number of all of the steps.
    n = len(partial_graphs)
    assert len(focus_nodes) == n, "Must be the same number of focus nodes as partial graphs."
    assert (
        len(correct_edge_choices) == n
    ), "Must have the same number of correct edge choices as partial graphs."
    assert (
        len(valid_edge_choices) == n
    ), "Must have the same number of valid edge choices as partial graphs."
    assert (
        len(graph_distance_to_target) == n
    ), "Must have the same number of graph distances as partial graphs."

    return GraphTraceSample(
        full_graph=graph_sample,
        partial_graphs=partial_graphs,
        focus_nodes=focus_nodes,
        correct_edge_choices=correct_edge_choices,
        valid_edge_choices=valid_edge_choices,
        distance_to_target=graph_distance_to_target,
        correct_attachment_point_choices=[None for _ in range(len(partial_graphs))],
        valid_attachment_point_choices=[[] for _ in range(len(partial_graphs))],
        correct_node_type_choices=[[] for _ in range(len(partial_graphs))],
        correct_first_node_type_choices=[],
    )


def node_idx_if_undiscovered(edge: Edge, node_states: Dict[int, NodeState]):
    """Given a typed edge, returns the node index of either the source or target if that node has
    not been discovered yet, as specified by the node_states dict.

    It will also change the state of the returned node so that it is equal to NodeState.DISCOVERED.

    Example use:
    >>> node_states = {0: NodeState.FOCUS, 1: NodeState.UNDISCOVERED, 2: NodeState.UNDISCOVERED}
    >>> edge = Edge(source=1, target=0, type=0)
    >>> node_idx_if_undiscovered(edge, node_states)
    1
    >>> assert node_states == {0: NodeState.FOCUS, 1: NodeState.DISCOVERED, 2: NodeState.UNDISCOVERED}
    >>> assert node_idx_if_undiscovered(edge, node_states) is None  # Node has been discovered now.

    Args:
        edge: a typed edge representing an edge proposed to be added to a graph.
        node_states: the current node state dictionary.

    Returns:
        The index of either the source or target of the edge

    """
    source = edge.source
    target = edge.target
    assert not any(
        node_states[idx] == NodeState.LOCKED for idx in (source, target)
    ), "We should never see an edge to a locked node."
    for idx in (source, target):
        if node_states[idx] == NodeState.UNDISCOVERED:
            node_states[idx] = NodeState.DISCOVERED
            return idx
    return None


def get_node_edges(
    node_idx_to_neighbours: Dict[int, EdgeCollection],
    focus_node_idx: int,
    node_states: Dict[int, NodeState],
) -> EdgeCollection:
    """Get the edges which have one end at the given focus node from the typed adjacency list,
    restricted to those nodes that are already marked as LOCKED (i.e., not visited by the
    generation).

    Example use:
    >>> adjacency_dict = {
    ...     0: [Edge(source=0, target=1, type=0)],
    ...     1: [Edge(source=1, target=0, type=1),
    ...         Edge(source=1, target=2, type=0),
    ...         Edge(source=1, target=3, type=2)],
    ...     2: [Edge(source=2, target=3, type=0)],
    ...     }
    ...
    >>> node_states = {
    ...     0: NodeState.LOCKED,
    ...     1: NodeState.FOCUS,
    ...     2: NodeState.DISCOVERED,
    ...     3: NodeState.UNDISCOVERED,
    ... }
    >>> result = get_node_edges(adjacency_dict, 1, node_states)
    >>> print(result)
    [Edge(source=1, target=2, type=0), Edge(source=1, target=3, type=2)]

    Args:
        node_idx_to_neighbours: A dictionary with integer keys representing node indices, and
            values equal to the neighbors of those keys.
        focus_node_idx: The index of the node for which we want the edges.
        node_states: a dictionary of int to NodeState, giving the states of nodes in the current
            step of exploration.

    Returns:
        A list of Edges which are have one end at the given focus node, are existant in the given
        adjacency list, and do not point to a locked node.

    """
    candidate_edges = node_idx_to_neighbours[focus_node_idx]
    unlocked_edges = [
        typed_edge
        for typed_edge in candidate_edges
        if node_states[typed_edge.target] is not NodeState.LOCKED
    ]
    return unlocked_edges


def get_open_edges(focus_node_idx: int, node_states: Dict[int, NodeState]) -> EdgeCollection:
    """Get the edges which have one end at the given focus node and the other end at a non-closed
    node.

    Example use:
    >>> node_states = {
    ...     0: NodeState.LOCKED,
    ...     1: NodeState.FOCUS,
    ...     2: NodeState.DISCOVERED,
    ...     3: NodeState.UNDISCOVERED,
    ...     4: NodeState.UNDISCOVERED,
    ... }
    >>> result = get_open_edges(1, node_states)
    >>> print(result)
    [Edge(source=1, target=2, type=-1), Edge(source=1, target=3, type=-1), Edge(source=1, target=4, type=-1)]

    Args:
        focus_node_idx: The index of the node for which we want the edges.
        node_states: a dictionary of int to NodeState, giving the states of nodes in the current
            step of exploration. The `get_edges` function will not return any edges to nodes which
            are in a LOCKED state.

    Returns:
        A list of Edges which are have one end at the given focus node, and the other end at
        any node which is not locked (or the focus node). Note: there is no valid type information
        stored in these edges!

    """
    unlocked_nodes = [
        node
        for node, state in node_states.items()
        if state in {NodeState.UNDISCOVERED, NodeState.DISCOVERED}
    ]
    unlocked_edges = [
        Edge(source=focus_node_idx, target=target, type=-1) for target in unlocked_nodes
    ]
    return unlocked_edges


def calculate_source_to_targets_dict(adjacency_list: AdjacencyList) -> Dict[int, EdgeCollection]:
    """Convert an of adjacency list into a dict from source node to EdgeCollection.

    Example use:
    >>> adjacency_list = [
    ...     Edge(source=0, target=1, type=0), Edge(source=1, target=2, type=0),
    ...     Edge(source=1, target=3, type=1),
    ...     ]
    >>> calculate_source_to_targets_dict(adjacency_list)
    defaultdict(<class 'list'>, {0: [Edge(source=0, target=1, type=0)], 1: [Edge(source=1, target=2, type=0), Edge(source=1, target=3, type=1)]})

    Args:
        adjacency_list: An AdjacencyList object representing the graph connections.

    Returns:
        A dictionary from the source node index to the collection of edges coming from that node.
    """
    adjacency_dict: Dict[int, EdgeCollection] = defaultdict(list)
    for edge in adjacency_list:
        adjacency_dict[edge.source].append(edge)
    return adjacency_dict


def __symmetrize_adjacency_list(adjacency_list):
    new_adjacency_list = []
    for edge in adjacency_list:
        new_adjacency_list.append(edge)
        if edge.source != edge.target:
            new_adjacency_list.append(Edge(source=edge.target, target=edge.source, type=edge.type))
    return new_adjacency_list


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
