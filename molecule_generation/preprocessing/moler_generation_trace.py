"""Function (and helpers) to map a GraphSample to a GraphTraceSample."""
import random
from collections import defaultdict
from copy import deepcopy
from enum import Enum
from queue import Queue
from typing import Any, Dict, List, Union, Set, Optional, Tuple, Type

import numpy as np
from rdkit.Chem import BondType, MolFromSmiles, RWMol

from molecule_generation.preprocessing.generation_order import GenerationOrder, BFSOrder
from molecule_generation.preprocessing.graph_sample import (
    AdjacencyList,
    Edge,
    GraphSample,
    GraphTraceSample,
)
from molecule_generation.chem.motif_utils import MotifAnnotation, MotifAtomAnnotation
from molecule_generation.chem.rdkit_helpers import initialise_atom_from_symbol
from molecule_generation.chem.valence_constraints import _max_valence_for_single_node


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


def graph_sample_to_MoLeR_trace(
    graph_sample: GraphSample, generation_order_cls: Type[GenerationOrder] = BFSOrder
) -> GraphTraceSample:
    """Randomly generate a GraphTraceSample from a single GraphSample.

    Note:
    We assume, for now, that the GraphSample contains only undirected edges. Practically, we take
    that assumption to mean that for each edge (u, v), there is _exactly one_ representative in the
    GraphSample adjacency lists. So, if (u, v) is in the adjacency lists, then (v, u) is not.

    Example use:
    >>> adjacency_list = [Edge(source=0, target=1, type=1),
    ...                   Edge(source=0, target=2, type=0),
    ...                   Edge(source=1, target=2, type=0)]
    >>> graph_sample = GraphSample(adjacency_list=adjacency_list,
    ...                            num_edge_types=4,
    ...                            node_features=[0., 0., 0.],
    ...                            graph_properties={"sa_score": 3.44},
    ...                            node_types=["C", "C", "O"],
    ...                            smiles_string="C1=CO1",
    ...                            )
    ...
    >>> graph_trace = graph_sample_to_MoLeR_trace(graph_sample)
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

    enclosing_motif_id = [None] * num_nodes
    for motif_id, motif in enumerate(graph_sample.motifs):
        for atom in motif.atoms:
            enclosing_motif_id[atom.atom_id] = motif_id

    generation_order = generation_order_cls(
        mol=graph_sample.mol or MolFromSmiles(graph_sample.smiles_string),
        enclosing_motif_id=enclosing_motif_id,
    )

    (
        exploration_order,
        valid_start_node_choices,
        all_valid_next_node_choices,
    ) = _construct_exploration_order(
        adjacency_list=graph_sample.adjacency_list,
        num_nodes=num_nodes,
        motifs=graph_sample.motifs,
        generation_order=generation_order,
        enclosing_motif_id=enclosing_motif_id,
    )

    # We renumber the nodes in the order they appear in `exploration_order`.
    old_id_to_new_id_mapping = _get_old_id_to_new_id_mapping(exploration_order)

    node_types = reindex_list(graph_sample.node_types, exploration_order)
    re_indexed_adjacency_list = reindex_adjacency_list(
        graph_sample.adjacency_list, exploration_order
    )

    valid_start_node_choices = [old_id_to_new_id_mapping[idx] for idx in valid_start_node_choices]

    all_valid_next_node_choices = [
        [old_id_to_new_id_mapping[idx] for idx in valid_next_node_choices]
        for valid_next_node_choices in all_valid_next_node_choices
    ]

    motifs = reindex_motifs(graph_sample.motifs, exploration_order)
    enclosing_motif_id = reindex_list(enclosing_motif_id, exploration_order)

    # Initialise the empty RWMol:
    mol = RWMol()

    # This is where the undirected assumption on the `graph_sample` is used.
    symmetrized_adj_list = __symmetrize_adjacency_list(re_indexed_adjacency_list)

    # Calculate a dictionary of node_idx -> list of neighboring nodes.
    node_idx_to_neighbours = calculate_source_to_targets_dict(symmetrized_adj_list)

    # We now build the target graph one node/edge at a time, recording each
    # intermediate step. We record our progress in these growing lists of
    # already created nodes/edges:
    cur_partial_node_list: List[int] = []
    cur_partial_adjacency_list: AdjacencyList = []

    # We keep the history of the partial node lists and adjacency lists, such that
    # the i-th partial graph is described by their i-th entries:
    partial_node_lists: List[List[int]] = []
    partial_adjacency_lists: List[AdjacencyList] = []

    # At each step in the graph generation, we keep the edges that we _could_ have chosen.
    # correct_edge_choices[i] will be the collection of all edges we could have added to
    # the adjacency list in partial_adjacency_lists[i] to get a correct partial adjacency list in
    # partial_adjacency_lists[i+1]
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

    # Here we record the correct node types for the next focus node when it gets updated.
    # Note: This will contain None for partial steps that do not choose a node (i.e., those
    # where we make an edge decision) and it will be empty exactly once, in the final step.
    correct_node_type_choices: List[Optional[List[str]]] = []

    # Here we record the correct attachment point for steps that attach a motif.
    correct_attachment_point_choices: List[Optional[int]] = []

    # We also keep track of valid attachment point choices.
    valid_attachment_point_choices: List[List[int]] = []

    # For next node type prediction, nodes inside motifs should be marked with the motif type.
    node_types_for_next_type_prediction = deepcopy(node_types)

    # Gather the symmetry class for motif nodes.
    node_symmetry_class = [None] * num_nodes

    for motif in motifs:
        for atom in motif.atoms:
            node_types_for_next_type_prediction[atom.atom_id] = motif.motif_type
            node_symmetry_class[atom.atom_id] = atom.symmetry_class_id

    correct_first_node_type_choices: List[str] = [
        node_types_for_next_type_prediction[node_idx] for node_idx in valid_start_node_choices
    ]

    def add_atom(atom_idx: int):
        # Add the atom to the molecule.
        mol.AddAtom(initialise_atom_from_symbol(node_types[atom_idx]))

        # Add to partial node list.
        cur_partial_node_list.append(atom_idx)

    def add_edge(edge: Edge):
        # Add the edge to the molecule.
        mol.AddBond(edge.source, edge.target, EDGE_TYPE_IDX_TO_RDKIT_TYPE[edge.type])

        # Add to partial adjacency list.
        cur_partial_adjacency_list.append(edge)

    def edge_inside_motif(edge: Edge):
        return equal_and_not_none(enclosing_motif_id[edge.source], enclosing_motif_id[edge.target])

    def add_generation_step(
        step_focus_node: int,
        step_correct_node_type_choices: Optional[List[str]],
        step_correct_edge_choices: EdgeCollection,
        step_valid_edge_choices: EdgeCollection,
        step_correct_attachment_point_choice: Optional[int],
        step_valid_attachment_point_choices: List[int],
    ):
        focus_nodes.append(step_focus_node)
        correct_node_type_choices.append(step_correct_node_type_choices)

        # Record all possible edges we could add at this step:
        correct_edge_choices.append(deepcopy(step_correct_edge_choices))

        # Record all valid open edges for this step:
        valid_edge_choices.append(deepcopy(step_valid_edge_choices))

        # Record the correct attachment point choice at this step:
        correct_attachment_point_choices.append(step_correct_attachment_point_choice)

        # Record all motif attachment point choices for this step:
        valid_attachment_point_choices.append(step_valid_attachment_point_choices)

        graph_distance_to_target.append(
            calculate_dist_from_focus_to_valid_target(
                cur_partial_adjacency_list, step_focus_node, node_states, step_valid_edge_choices
            )
        )

        molecule_trace.append(RWMol(mol))

        # Record the current state of the partial graph:
        partial_node_lists.append(deepcopy(cur_partial_node_list))
        partial_adjacency_lists.append(deepcopy(cur_partial_adjacency_list))

    # Each motif is handled when we see its first node, which is the attachment point.
    seen_motif_ids: Set[int] = set()

    for focus_node_idx, valid_next_node_choices in enumerate(all_valid_next_node_choices):
        current_motif_id = enclosing_motif_id[focus_node_idx]

        if current_motif_id is None:
            add_atom(focus_node_idx)
        elif current_motif_id not in seen_motif_ids:
            # New motif; add the entire motif into the partial graph.
            nodes_to_add = sorted([atom.atom_id for atom in motifs[current_motif_id].atoms])

            # The first motif node in `exploration_order` should be the attachment point.
            assert focus_node_idx == nodes_to_add[0]

            valid_attachment_points = []
            seen_symmetry_classes = set()

            for node_idx in nodes_to_add:
                add_atom(node_idx)

                if node_symmetry_class[node_idx] not in seen_symmetry_classes:
                    seen_symmetry_classes.add(node_symmetry_class[node_idx])
                    valid_attachment_points.append(node_idx)

            for node_idx in nodes_to_add:
                node_states[node_idx] = NodeState.LOCKED

                for edge in node_idx_to_neighbours[node_idx]:
                    # Make sure we don't add the same edge twice.
                    if (
                        enclosing_motif_id[edge.target] == current_motif_id
                        and edge.target < node_idx
                    ):
                        add_edge(edge)

            # Mark this motif as already seen.
            seen_motif_ids.add(current_motif_id)

            open_attachment_points = get_open_attachment_points(
                valid_attachment_points, cur_partial_adjacency_list, node_types
            )

            # Pick the attachment point, but only if the current partial graph is non-empty, and
            # there are at least two candidates to pick from.
            if focus_node_idx > 0 and len(open_attachment_points) > 1:
                # Now add a step to predict the attachment point. Note that the focus node part here is
                # tricky: currently, all steps have a focus node, and to avoid changing that we set the
                # attachment point as the focus node. However, we don't want the attachment point
                # selection step to know the ground-truth attachment point! But, for the attachment
                # point selection step, we will later mark *all* possible choices with the focus node
                # bit, to make it explicit which nodes will be attachment point candidates, but also as
                # a side-effect we will "hide" the ground-truth answer from the model.
                add_generation_step(
                    step_focus_node=focus_node_idx,
                    step_correct_node_type_choices=None,
                    step_correct_edge_choices=[],
                    step_valid_edge_choices=[],
                    step_correct_attachment_point_choice=focus_node_idx,
                    step_valid_attachment_point_choices=open_attachment_points,
                )
        else:
            # Motif already processed, nothing to do here.
            continue

        node_states[focus_node_idx] = NodeState.FOCUS
        edges_to_add = get_node_edges(node_idx_to_neighbours, focus_node_idx, node_states)
        edges_to_open_nodes = get_open_edges(focus_node_idx, node_states)

        # Edges inside motifs should neither be added nor even considered.
        edges_to_add = [edge for edge in edges_to_add if not edge_inside_motif(edge)]
        edges_to_open_nodes = [edge for edge in edges_to_open_nodes if not edge_inside_motif(edge)]

        if focus_node_idx > 0 and current_motif_id is not None:
            # Motifs should be connected to a non-empty partial graph with exactly one edge.
            assert len(edges_to_add) == 1

        random.shuffle(edges_to_add)
        while edges_to_add:
            add_generation_step(
                step_focus_node=focus_node_idx,
                step_correct_node_type_choices=None,
                step_correct_edge_choices=edges_to_add,
                step_valid_edge_choices=edges_to_open_nodes,
                step_correct_attachment_point_choice=None,
                step_valid_attachment_point_choices=[],
            )

            # Get the edge that we actually do add:
            edge = edges_to_add.pop()

            # Remove that edge from the 'edges_to_open_nodes' list.
            edges_to_open_nodes.remove(Edge(source=edge.source, target=edge.target, type=-1))

            add_edge(edge)

        # Add a step where there are no correct edge choices, so that we can learn what these look
        # like as well.
        add_generation_step(
            step_focus_node=focus_node_idx,
            step_correct_node_type_choices=[
                node_types_for_next_type_prediction[node_idx]
                for node_idx in valid_next_node_choices
            ],
            step_correct_edge_choices=[],
            step_valid_edge_choices=edges_to_open_nodes,
            step_correct_attachment_point_choice=None,
            step_valid_attachment_point_choices=[],
        )

        node_states[focus_node_idx] = NodeState.LOCKED

    # Make sure next node type choices look as expected: a non-empty list throughout the process,
    # except for the last step, where the list should be empty.
    assert not any(choices == [] for choices in correct_node_type_choices[:-1])
    assert not correct_node_type_choices[-1]

    node_features = reindex_list(graph_sample.node_features, exploration_order)

    if graph_sample.node_categorical_features is not None:
        node_categorical_features = reindex_list(
            graph_sample.node_categorical_features, exploration_order
        )
    else:
        node_categorical_features = None

    # As we changed the order of nodes, we need to make sure that our graph sample
    # matches that:
    reindexed_graph_sample = GraphSample(
        adjacency_list=re_indexed_adjacency_list,
        num_edge_types=graph_sample.num_edge_types,
        node_features=node_features,
        node_categorical_features=node_categorical_features,
        node_categorical_num_classes=graph_sample.node_categorical_num_classes,
        graph_properties=graph_sample.graph_properties,
        node_types=node_types,
        smiles_string=graph_sample.smiles_string,
        mol=RWMol(mol),
        motifs=motifs,
    )

    result = convert_lists_to_graph_trace_sample(
        reindexed_graph_sample,
        partial_node_lists,
        partial_adjacency_lists,
        correct_edge_choices,
        valid_edge_choices,
        correct_attachment_point_choices,
        valid_attachment_point_choices,
        focus_nodes,
        graph_distance_to_target,
        molecule_trace,
        correct_node_type_choices,
        correct_first_node_type_choices,
    )
    return result


def get_open_attachment_points(
    valid_attachment_points: List[int], adjacency_list: AdjacencyList, node_types: List[str]
) -> List[int]:
    """Filter down attachment points to those that still have some valence left.

    Args:
        valid_attachment_points: the list of valid attachment points to filter down.
        adjacency_list: the adjacency list representation of the graph.
        node_types: a list of node types (e.g. ["C", "O", "C"]).

    Returns:
        Valid attachment points, filtered down to those that still have valence left.

    """
    node_idx_to_valency_map = defaultdict(int)

    for edge in adjacency_list:
        node_idx_to_valency_map[edge.source] += edge.type + 1
        node_idx_to_valency_map[edge.target] += edge.type + 1

    open_attachment_points = []

    for node_idx in valid_attachment_points:
        if node_idx_to_valency_map[node_idx] < _max_valence_for_single_node(node_types[node_idx]):
            open_attachment_points.append(node_idx)

    return open_attachment_points


def equal_and_not_none(obj_1: Any, obj_2: Any) -> bool:
    return obj_1 == obj_2 and obj_1 is not None


def _construct_exploration_order(
    adjacency_list: Union[EdgeCollection, List[np.ndarray]],
    num_nodes: int,
    motifs: List[MotifAnnotation],
    enclosing_motif_id: List[Optional[int]],
    generation_order: GenerationOrder,
    symmetrise_adjacency_list: bool = True,
) -> Tuple[List[int], List[int], List[List[int]]]:
    """Compute a BFS order of the nodes given the adjacency lists.

    Args:
        adjacency_list: the adjacency list representation of the graph.
        num_nodes: the number of nodes in the current graph.
        motifs: a list of motifs present in the graph.
        enclosing_motif_id: a list containing the enclosing motif id for each node covered by a
            motif, or None for nodes that are not part of any motif.
        generation_order: object that defines in which order to pick the nodes.
        symmetrise_adjacency_list: bool representing whether the adjacency list provided needs to be
            symmetrised, or already is. The main body of this function requires that the adjacency
            list is symmetric: if it contains the edge (u, v), it must also contain (v, u). If that
            is already the case, this parameter should be false. If is not the case, and the
            adjacency list contains only one representative of (u, v) and (v, u), then this
            parameter should be true.

    Returns:
        A tuple containing:
            - an ordering of node ids, defined by the graph and given `GenerationOrder`
            - list of valid first node candidates
            - list of valid next node candidates in each generation step

    """
    if len(adjacency_list) == 0:
        assert num_nodes == 1
        return [0]

    if isinstance(adjacency_list, np.ndarray) or isinstance(adjacency_list[0], np.ndarray):
        adjacency_list = _convert_np_to_edgecollection(adjacency_list)

    if symmetrise_adjacency_list:
        adjacency_list = __symmetrize_adjacency_list(adjacency_list)

    node_idx_to_neighbours = calculate_source_to_targets_dict(adjacency_list)

    # Pick the first node according to the generation order.
    valid_start_node_choices = generation_order.get_valid_start_node_choices()
    starting_node_idx = generation_order.pick_start_node(valid_start_node_choices)

    exploration_queue = [starting_node_idx]
    queued_nodes = {starting_node_idx}

    # Exploration order, inner lists are "chunks" (motifs). This gets flattened later.
    exploration_order: List[List[int]] = []
    all_valid_next_node_choices: List[List[int]] = []

    # Search of the graph, defined by the adjacency list and generation order.
    while exploration_queue:
        # Select valid node choices. We use a copy of the exploration queue, so that we're safe in
        # the case when the list we get back is the same _instance_.
        valid_next_node_choices = generation_order.get_valid_next_node_choices(
            deepcopy(exploration_queue), sum(exploration_order, [])
        )

        current_node = generation_order.pick_next_node(valid_next_node_choices)

        exploration_queue.remove(current_node)

        current_node_motif = enclosing_motif_id[current_node]

        if current_node_motif is None:
            new_nodes = [current_node]
        else:
            # Gather motif nodes, but make sure the attachment point is the first one.
            new_nodes = [current_node]

            for atom in motifs[current_node_motif].atoms:
                atom_id = atom.atom_id

                if atom_id != current_node:
                    new_nodes.append(atom_id)

            # The other motif nodes were never queued, and they don't need to be.
            queued_nodes.update(new_nodes[1:])

        # Note these two lists are shifted: `valid_next_node_choices` should be predicted in the
        # _previous_ generation step. We will allign them later.
        exploration_order.append(new_nodes)
        all_valid_next_node_choices.append(valid_next_node_choices)

        for node in new_nodes:
            neighbour_nodes = [edge.target for edge in node_idx_to_neighbours[node]]

            for neighbour in neighbour_nodes:
                if neighbour not in queued_nodes:
                    queued_nodes.add(neighbour)
                    exploration_queue.append(neighbour)

    # Strip the first step; we will handle the first node type separately. This also makes the
    # exploration order and the valid node choices alligned.
    all_valid_next_node_choices = all_valid_next_node_choices[1:]

    # In the last step, there are no new nodes to add.
    all_valid_next_node_choices.append([])

    # Make sure the lengths match.
    assert len(exploration_order) == len(all_valid_next_node_choices)

    # Flatten the `exploration_order` and fill up missing values in `all_valid_next_node_choices`.
    exploration_order_final: List[int] = []
    all_valid_next_node_choices_final: List[List[int]] = []

    for new_nodes, valid_next_node_choices in zip(exploration_order, all_valid_next_node_choices):
        exploration_order_final += new_nodes
        all_valid_next_node_choices_final.append(valid_next_node_choices)
        all_valid_next_node_choices_final.extend([[] for _ in range(len(new_nodes) - 1)])

    # The lengths should still match.
    assert len(exploration_order_final) == len(all_valid_next_node_choices_final)

    return exploration_order_final, valid_start_node_choices, all_valid_next_node_choices_final


def reindex_list(lst: List, exploration_order: List[int]) -> List:
    """Change the order of a list so that it matches the exploration order.

    Example use:
        >>> node_types = ["C", "N", "F", "Br"]
        >>> exploration_order = [1, 0, 3, 2]
        >>> reindex_list(node_types, exploration_order)
        ['N', 'C', 'Br', 'F']

    """
    return [lst[idx] for idx in exploration_order]


def _get_old_id_to_new_id_mapping(exploration_order: List[int]) -> List[int]:
    return {old_idx: new_idx for new_idx, old_idx in enumerate(exploration_order)}


def reindex_motifs(
    motifs: List[MotifAnnotation], exploration_order: List[int]
) -> List[MotifAnnotation]:
    """Update the motif list to match the exploration order."""

    idx_map = _get_old_id_to_new_id_mapping(exploration_order)
    reindexed_motifs = []

    for motif in motifs:
        reindexed_atoms = []

        for atom in motif.atoms:
            reindexed_atoms.append(
                MotifAtomAnnotation(
                    atom_id=idx_map[atom.atom_id], symmetry_class_id=atom.symmetry_class_id
                )
            )

        reindexed_motifs.append(MotifAnnotation(motif_type=motif.motif_type, atoms=reindexed_atoms))

    return reindexed_motifs


def reindex_adjacency_list(adjacency_list: List[Edge], exploration_order: List[int]) -> List[Edge]:
    """Update the given adjacency list so that the edges match the reindexing implied by the
    exploration order.

    Example use:
        >>> adjacency_list = [Edge(0, 1, 1), Edge(0, 2, 1), Edge(3, 2, 2)]
        >>> exploration_order = [1, 3, 0, 2]
        >>> reindex_adjacency_list(adjacency_list, exploration_order)
        [Edge(source=2, target=0, type=1), Edge(source=2, target=3, type=1), Edge(source=1, target=3, type=2)]
    """
    idx_map = _get_old_id_to_new_id_mapping(exploration_order)
    return [
        Edge(source=idx_map[e.source], target=idx_map[e.target], type=e.type)
        for e in adjacency_list
    ]


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
    adjacency_list: Union[EdgeCollection, List[np.ndarray]],
    node_subset: Union[List[int], int],
    node_states: Dict[int, NodeState],
    valid_edges: Union[EdgeCollection, np.ndarray],
    symmetrise_adjacency_list: bool = True,
    valid_target_node_states: Optional[Set[NodeState]] = None,
) -> List[int]:
    """Calculate the graph distance between the focus node and the targets of the valid edges.

    Args:
        adjacency_list: the adjacency list representation of the graph.
        node_subset: the index of the node to which we want the distances calculated.
        node_states: the node states of the current graph.
        valid_edges: the list of valid edges we are considering at this stage. The target nodes of
            all of the edges in this list should equal the valid_target_set exactly. All the source
            nodes should be equal to the focus node.
        symmetrise_adjacency_list: bool representing whether the adjacency list provided needs to be symmetrised, or
            already is. The main body of this function requires that the adjacency list is symmetric: if it contains the
            edge (u, v), it must also contain (v, u). If that is already the case, this parameter should be false. If
            is not the case, and the adjacency list contains only one representative of (u, v) and (v, u), then this
            parameter should be true.
        valid_target_node_states: a set which gives the node states for which we will allow edges.

    Returns:
        The graph distance between the focus node and the target of each of the valid edges. The
        order between these distances and the valid edges is preserved! So, if the edge
        valid_edges[i] has target node v, then the distance between the focus node and v is in index
        i of the returned list.

    Note:
        Unconnected nodes will have a distance of `0`

    """
    # Avoid mutable default argument.
    if valid_target_node_states is None:
        valid_target_node_states = {NodeState.DISCOVERED, NodeState.LOCKED}

    # Easy out for the degenerate case:
    if len(valid_edges) == 0:
        return []

    if len(adjacency_list) == 0:
        return [0] * len(valid_edges)

    if isinstance(adjacency_list, np.ndarray) or isinstance(adjacency_list[0], np.ndarray):
        adjacency_list = _convert_np_to_edgecollection(adjacency_list)

    if isinstance(valid_edges, np.ndarray):
        valid_edges = _convert_np_to_edgecollection(valid_edges)

    if symmetrise_adjacency_list:
        adjacency_list = __symmetrize_adjacency_list(adjacency_list)
    node_idx_to_neighbours = calculate_source_to_targets_dict(adjacency_list)
    # There are fewer nodes to explore than edges, so the maxsize below is certainly enough.
    if isinstance(node_subset, int):
        node_subset = [node_subset]
    exploration_queue = Queue(maxsize=len(adjacency_list))
    distance_dict = {}
    for node_idx in node_subset:
        exploration_queue.put(node_idx)
        distance_dict[node_idx] = 0

    # This is used for an early stopping criteria in the BFS. Since all of the undiscovered nodes
    # are not yet connected to the graph given by the adjacency list in this function, we will not
    # find them in the BFS.
    valid_target_set = {
        edge.target for edge in valid_edges if node_states[edge.target] in valid_target_node_states
    }

    # Breadth first search of the graph defined by the given adjacency_list
    while len(valid_target_set) > 0:
        if exploration_queue.empty():
            break
        current_node = exploration_queue.get()
        neighbour_nodes = [edge.target for edge in node_idx_to_neighbours[current_node]]

        for neighbour in neighbour_nodes:
            if neighbour not in distance_dict:
                exploration_queue.put(neighbour)
                distance_dict[neighbour] = distance_dict[current_node] + 1
                valid_target_set.discard(neighbour)

    # Note: Unconnected nodes will have a distance of `0`.
    distance_to_targets = [distance_dict.get(edge.target, 0) for edge in valid_edges]
    return distance_to_targets


def convert_lists_to_graph_trace_sample(
    graph_sample: GraphSample,
    partial_node_lists: List[List[int]],
    partial_adjacency_lists: List[AdjacencyList],
    correct_edge_choices: List[EdgeCollection],
    valid_edge_choices: List[EdgeCollection],
    correct_attachment_point_choices: List[Optional[int]],
    valid_attachment_point_choices: List[List[int]],
    focus_nodes: List[int],
    graph_distance_to_target: List[List[int]],
    molecule_trace: List[RWMol],
    correct_node_type_choices: List[Optional[List[str]]],
    correct_first_node_type_choices: List[str],
) -> GraphTraceSample:
    """Convert various generation trace related lists to a single GraphTraceSample object.

    Args:
        graph_sample: the original GraphSample object that the generation trace is made from.
        partial_adjacency_lists: the sequence of adjacency lists that make up the generation trace.
        correct_edge_choices: the collection of all correct edges at each stage in the generation
            trace. For example, correct_edge_choices[i] should contain all the edges in graph_sample
            that are connected to focus_nodes[i] which are _not_ already in partial_adjacency_lists[i].
        valid_edge_choices: the collection of all possible edge choices at each stage in the
            generation trace. A possible edge choice is defined as an edge starting at the focus
            node and ending at a node which is not locked (or the focus node).
        correct_attachment_point_choices: contains the correct attachment point choice for those
            steps in the generation trace that require picking one, and `None` otherwise.
        valid_attachment_point_choices: contains the collection of all candidate attachment point
            choices for each generation step. For steps which do not require picking an attachment
            point, this collection will be empty.
        focus_nodes: a list containing the index of the focus node at each step in the generation
            trace.
        graph_distance_to_target: a list of lists, each containing the graph distance between the
            focus node and the valid edge choice targets at each step of the generation trace.
        molecule_trace: a list of RWMol objects which mirror the generation trace of the partial graphs.
        correct_node_type_choices: a list of lists, each containing the correct node types if the index
            represents a "node choice" step. Otherwise it will be an empty list.
        correct_first_node_type_choices: a list of node types that would be a valid first node type.

    Returns:
        A GraphTraceSample object containing all of the information of this particular graph
        generation.

    """
    # Make sure we have not made an error in the generation/molecule trace:
    assert len(partial_adjacency_lists) == len(
        molecule_trace
    ), "List of partial adjacency lists and molecule trace are different lengths."
    assert len(partial_adjacency_lists) == len(
        partial_node_lists
    ), "List of partial adjacency lists and partial node lists are different lengths."
    node_features = graph_sample.node_features
    graph_properties = graph_sample.graph_properties
    smiles_string = graph_sample.smiles_string
    num_edge_types = graph_sample.num_edge_types
    node_types = graph_sample.node_types
    partial_graphs = [
        GraphSample(
            adjacency_list=partial_adj_list,
            num_edge_types=num_edge_types,
            node_features=[node_features[node_idx] for node_idx in partial_node_list],
            graph_properties=graph_properties,
            node_types=node_types,
            smiles_string=smiles_string,
            mol=mol,
        )
        for partial_node_list, partial_adj_list, mol in zip(
            partial_node_lists, partial_adjacency_lists, molecule_trace
        )
    ]

    # Double check nodes and edges in partial graphs.
    for partial_node_list, partial_adj_list, mol in zip(
        partial_node_lists, partial_adjacency_lists, molecule_trace
    ):
        num_nodes = len(partial_node_list)

        assert partial_node_list == list(
            range(num_nodes)
        ), "Nodes belonging to a partial graph are not consecutive."

        assert (
            len(mol.GetAtoms()) == num_nodes
        ), "Number of atoms in a partial graph does not match number of graph nodes."

        for edge in partial_adj_list:
            for node in [edge.source, edge.target]:
                assert node < num_nodes, "Partial graph contains an invalid edge."

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
        len(correct_attachment_point_choices) == n
    ), "Must have the same number of correct attachment point choices as partial graphs."
    assert (
        len(valid_attachment_point_choices) == n
    ), "Must have the same number of valid attachment point choices as partial graphs."
    assert (
        len(graph_distance_to_target) == n
    ), "Must have the same number of graph distances as partial graphs."
    assert (
        len(correct_node_type_choices) == n
    ), "Must have the same number of correct node type entries as partial graphs."

    return GraphTraceSample(
        full_graph=graph_sample,
        partial_graphs=partial_graphs,
        focus_nodes=focus_nodes,
        correct_edge_choices=correct_edge_choices,
        valid_edge_choices=valid_edge_choices,
        distance_to_target=graph_distance_to_target,
        correct_attachment_point_choices=correct_attachment_point_choices,
        valid_attachment_point_choices=valid_attachment_point_choices,
        correct_node_type_choices=correct_node_type_choices,
        correct_first_node_type_choices=correct_first_node_type_choices,
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
    restricted to those nodes that are already marked as LOCKED (i.e., visited by the generation).

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
    [Edge(source=1, target=0, type=1)]

    Args:
        node_idx_to_neighbours: A dictionary with integer keys representing node indices, and
            values equal to the neighbors of those keys.
        focus_node_idx: The index of the node for which we want the edges.
        node_states: a dictionary of int to NodeState, giving the states of nodes in the current
            step of exploration.

    Returns:
        A list of Edges which are have one end at the given focus node, are existant in the given
        adjacency list, and point to a locked node.

    """
    candidate_edges = node_idx_to_neighbours[focus_node_idx]
    locked_edges = [
        typed_edge
        for typed_edge in candidate_edges
        if node_states[typed_edge.target] is NodeState.LOCKED
    ]
    return locked_edges


def get_open_edges(focus_node_idx: int, node_states: Dict[int, NodeState]) -> EdgeCollection:
    """Get the edges which have one end at the given focus node and the other end at an
    already visited node.

    Example use:
    >>> node_states = {
    ...     0: NodeState.LOCKED,
    ...     1: NodeState.LOCKED,
    ...     2: NodeState.FOCUS,
    ...     3: NodeState.DISCOVERED,
    ...     4: NodeState.UNDISCOVERED,
    ... }
    >>> result = get_open_edges(2, node_states)
    >>> print(result)
    [Edge(source=2, target=0, type=-1), Edge(source=2, target=1, type=-1)]

    Args:
        focus_node_idx: The index of the node for which we want the edges.
        node_states: a dictionary of int to NodeState, giving the states of nodes in the current
            step of exploration.

    Returns:
        A list of Edges which are have one end at the given focus node, and the other end at
        any node which is locked (or the focus node). Note: there is no valid type information
        stored in these edges!

    """
    locked_nodes = [node for node, state in node_states.items() if state == NodeState.LOCKED]
    unlocked_edges = [
        Edge(source=focus_node_idx, target=target, type=-1) for target in locked_nodes
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
