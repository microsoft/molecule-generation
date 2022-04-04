"""Utility classes and methods for beam search."""
from copy import deepcopy
from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Set

import numpy as np
from rdkit.Chem import BondType, RWMol, Mol

from molecule_generation.chem.rdkit_helpers import initialise_atom_from_symbol
from molecule_generation.preprocessing.data_conversion_utils import remove_non_max_frags
from molecule_generation.preprocessing.cgvae_generation_trace import NodeState


class MoleculeGenerationEdgeCandidateInfo(NamedTuple):
    target_node_idx: int
    score: float
    logprob: float
    correct: Optional[bool]
    type_idx_to_logprobs: np.ndarray


class MoleculeGenerationStepInfo(NamedTuple):
    focus_node_idx: int
    partial_molecule_adjacency_lists: List[np.ndarray]
    candidate_edge_infos: List[MoleculeGenerationEdgeCandidateInfo]
    no_edge_score: float
    no_edge_logprob: float
    no_edge_correct: Optional[bool]


class Ray:
    def __init__(
        self,
        logprob: float,
        idx: int,
        adjacency_lists: List[np.ndarray],
        focus_node: int,
        node_states: Dict[int, NodeState],
        node_types: List[str],
        exploration_queue: List[int],
        generation_trace: Optional[List[MoleculeGenerationStepInfo]],
    ):
        """Ray constructor for beam search. Should usually be called from the `construct_ray` class method.

        Note: the adjacency_lists given must be symmetric. That is, for every edge (u, v) that they
            contain, they must also contain the edge (v, u).

            If the adjacency lists given have been symmetrised by the `construct_ray` method, this will be guaranteed.

        """
        # Sanity check.
        assert node_states[focus_node] == NodeState.FOCUS

        self.logprob = logprob
        self.idx = idx
        self.adjacency_lists = adjacency_lists
        self._node_idx_to_neighbours: Dict[int, Set] = {i: set() for i in range(len(node_states))}
        for adjacency_list in adjacency_lists:
            for edge in adjacency_list:
                source = edge[0]
                target = edge[1]
                self._node_idx_to_neighbours[source].add(target)

        # Need to convert from numpy int to python int for rdkit.
        self.focus_node = int(focus_node)
        self.node_states = node_states
        self.node_types = node_types
        self.exploration_queue = exploration_queue
        self.generation_trace = generation_trace
        self._finished = False

        # Initialise a molecule with no edges.
        self._molecule = RWMol()
        for node_symbol in node_types:
            self._molecule.AddAtom(initialise_atom_from_symbol(node_symbol))

        self._edge_type_idx_to_rdkit_type = {
            0: BondType.SINGLE,
            1: BondType.DOUBLE,
            2: BondType.TRIPLE,
        }

        # Add edges to the molecule.
        for edge_type_idx, adjacency_list in enumerate(adjacency_lists):
            edge_type = self._edge_type_idx_to_rdkit_type.get(edge_type_idx, None)

            if edge_type is None:
                break

            for source, target in adjacency_list:
                if source < target:
                    # Create python integers because RDKit doesn't like numpy.
                    self._molecule.AddBond(int(source), int(target), edge_type)

    @classmethod
    def construct_ray(
        cls,
        idx: int,
        focus_node: int,
        num_edge_types: int,
        node_types: List[str],
        add_self_loop_edges: bool,
        adjacency_lists: Optional[List[np.ndarray]] = None,
        connection_nodes: Optional[List[int]] = None,
        symmetrize_adjacency_lists: bool = True,
        store_generation_trace: bool = False,
    ):
        """Construct an empty ray with the given parameters.

        Args:
            idx: the index of the ray.
            focus_node: the index of the focus node to initialise the ray with.
            num_edge_types: the number of edge types in the graph.
            node_types: a list of string representations of the node types.
            add_self_loop_edges: indicates if we should add self-loop edges
            adjacency_lists: a list of numpy arrays corresponding to the adjacency lists with which we want to seed the
                the ray. All ray updates will build on these adjacency lists.
            connection_nodes: a subset of the nodes in the adjacency lists, to which we can connect new atoms.
            symmetrize_adjacency_lists: a bool representing whether we need to symmetrize the adjacency lists. If the
                adjacency lists are symmetric (i.e. for every edge (u, v) in the adjacency lists, the edge (v, u) is
                is also in the adjacency list), this should be false. Otherwise it must be true.
            store_generation_trace: bool flag indicating if all intermediate steps and decisions
                should be recorded; for example for visualisations and debugging purposes.

        Returns:
            A ray with all node states initialised to UNDISCOVERED except for the focus node, which
            has state FOCUS. All the of adjacency lists will be empty, as will the exploration
            queue.

        """
        if adjacency_lists is None:
            adjacency_lists = [
                np.zeros(shape=(0, 2), dtype=np.int32) for _ in range(num_edge_types)
            ]

        # We are about to mutate the adjacency lists. Make sure this does not escape from this function!
        adjacency_lists = list(adjacency_lists)

        num_nodes = len(node_types)
        if symmetrize_adjacency_lists:
            for i, adjacency_list in enumerate(adjacency_lists):
                reversed_adjacency_list = adjacency_list[:, ::-1]
                symmetrised_adjacency_list = np.empty(
                    shape=(2 * len(adjacency_list), 2), dtype=adjacency_list.dtype
                )
                # Interleave the edges and reversed edges.
                symmetrised_adjacency_list[0::2, :] = adjacency_list
                symmetrised_adjacency_list[1::2, :] = reversed_adjacency_list
                adjacency_lists[i] = symmetrised_adjacency_list

        if add_self_loop_edges:
            adjacency_lists.append(
                np.repeat(np.arange(num_nodes, dtype=np.int32), 2).reshape(-1, 2)
            )
        node_states = {node_idx: NodeState.UNDISCOVERED for node_idx in range(len(node_types))}

        # Lock all of the nodes that are present in the adjacency lists.
        for edge_type_idx in range(num_edge_types):
            adjacency_list = adjacency_lists[edge_type_idx]
            for edge in adjacency_list:
                source = edge[0]
                target = edge[1]
                node_states[source] = NodeState.LOCKED
                node_states[target] = NodeState.LOCKED

        # Unlock those edges in the scaffold to which we are allowed to connect.
        if connection_nodes is not None:
            for node_idx in connection_nodes:
                if node_states[node_idx] != NodeState.LOCKED:
                    print("WARNING! Adding a connection node which is not connected.")
                node_states[node_idx] = NodeState.DISCOVERED

        node_states[focus_node] = NodeState.FOCUS
        exploration_queue = [
            node_idx
            for node_idx, node_state in node_states.items()
            if node_state == NodeState.DISCOVERED
        ]
        return Ray(
            logprob=0.0,
            idx=idx,
            adjacency_lists=adjacency_lists,
            focus_node=focus_node,
            node_states=node_states,
            node_types=node_types,
            exploration_queue=exploration_queue,
            generation_trace=[] if store_generation_trace else None,
        )

    def __str__(self):
        return (
            f"=========================================\n"
            f"==== Probability\n"
            f"{np.exp(self.logprob)}\n"
            f"==== Adjacency lists\n"
            f"{self.adjacency_lists}\n"
            f"==== Node states\n"
            f"{self.node_states}\n"
            f"==== Focus node\n"
            f"{self.focus_node}\n"
        )

    @property
    def finished(self):
        return self._finished

    @property
    def molecule(self) -> RWMol:
        return deepcopy(self._molecule)

    @property
    def ro_molecule(self) -> Mol:
        mol = remove_non_max_frags(self._molecule)
        return Mol(mol)

    def update_focus_node(self):
        """Change the focus node to the next one in the exploration queue (if that exists)."""
        self.node_states[self.focus_node] = NodeState.LOCKED
        if len(self.exploration_queue) == 0:
            self._finished = True
            return

        # Need to convert from numpy int to python int, for rdkit.
        new_focus_node = int(self.exploration_queue.pop())
        self.focus_node = new_focus_node
        self.node_states[new_focus_node] = NodeState.FOCUS

    def add_edge(self, edge: np.ndarray, edge_type: int):
        """Add an edge to the graph in this ray of type edge_type.

        Args:
            edge: a numpy array of shape (2,) containing the [source, target] of the edge.
            edge_type: the edge type of the edge to be added.

        Returns:
            Nothing, but mutates the target node state from UNDISCOVERED to DISCOVERED and inserts
            the target into the exploration queue, if necessary.

        """
        edge_source = edge[0]
        edge_target = edge[-1]
        # Make sure we haven't made a mistake somewhere:
        assert self.node_states[edge_target] not in {
            NodeState.FOCUS,
            NodeState.LOCKED,
        }, "Tried to attach an edge to a node that is locked or is the current focus node."
        assert (
            self.node_states[edge_source] == NodeState.FOCUS
        ), "Tried to add an edge from a node that is not the focus node."
        assert (
            edge_target not in self._node_idx_to_neighbours[edge_source]
        ), "Tried to add an edge between connected nodes."

        # Add to the correct adjacency list.
        edge_type_adjacency_list = self.adjacency_lists[edge_type]
        # We tie the forwards and backwards edges here, so if edge (3, 5) is added to the list, then so is (5, 3).
        self.adjacency_lists[edge_type] = np.concatenate(
            (edge_type_adjacency_list, edge.reshape(-1, 2), edge[::-1].reshape(-1, 2))
        )

        # Add the edge to the molecule representation. Symmetrisation of adjacency list is done for us by RDKit.
        self._molecule.AddBond(
            edge_source.item(), edge_target.item(), self._edge_type_idx_to_rdkit_type[edge_type]
        )

        # Add the the neighbour dictionary.
        self._node_idx_to_neighbours[edge_source].add(edge_target)
        self._node_idx_to_neighbours[edge_target].add(edge_source)

        # Update the exploration queue and node states.
        if self.node_states[edge_target] == NodeState.UNDISCOVERED:
            self.node_states[edge_target] = NodeState.DISCOVERED
            self.exploration_queue.insert(0, edge_target)

    def contains_edge(self, source_idx: int, target_idx: int):
        """Returns true if the graph in this ray contains an edge from the given source to target."""
        return target_idx in self._node_idx_to_neighbours[source_idx]


class ExtensionType(Enum):
    ADD_EDGE = 0
    STOP_NODE = 1


class RayExtension(NamedTuple):
    logprob: float
    edge_choice: Optional[np.ndarray]
    edge_type: Optional[int]
    ray_idx: int  # The index of the beam to which this extension should be applied.
    extension_type: ExtensionType
    generation_step_info: MoleculeGenerationStepInfo
    type_logprob: float = 0


def extend_beam(extension_choices: List[RayExtension], beam: List[Ray]) -> List[Ray]:
    """Extend beam based on the supplied extension choices.

    Args:
        extension_choices: a list of RayExtension objects, each representing a choice of how to
            update the list of given rays.
        beam: a list of ray objects to be updated.

    Returns:
        A list of Ray objects that has the same length as the list of supplied extension_choices.
        The resulting list has the same order as the given extension_choices, which is not
        necessarily the same as that of the given list of rays.

    """
    new_beam: List[Ray] = [None] * len(extension_choices)
    for i, extension_choice in enumerate(extension_choices):
        new_beam[i] = deepcopy(beam[extension_choice.ray_idx])
        new_beam[i].idx = i
        new_beam[i].logprob = extension_choice.logprob
        if (
            new_beam[i].generation_trace is not None
            and extension_choice.generation_step_info is not None
        ):
            new_beam[i].generation_trace.append(extension_choice.generation_step_info)

        if extension_choice.extension_type == ExtensionType.STOP_NODE:
            # Move to the next focus node in the queue.
            new_beam[i].update_focus_node()

        else:
            # Add the chosen edge to the appropriate adjacency list.
            new_beam[i].add_edge(extension_choice.edge_choice, extension_choice.edge_type)
            # Correct the logprob to incorporate the edge type choice information.
            new_beam[i].logprob += extension_choice.type_logprob
    return new_beam
