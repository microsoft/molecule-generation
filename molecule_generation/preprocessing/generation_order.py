import random
from abc import abstractmethod
from typing import List, Union, Optional

from rdkit.Chem import GetSymmSSSR, Mol, RWMol

from molecule_generation.chem.rdkit_helpers import compute_canonical_atom_order


class GenerationOrder:
    def __init__(self, mol: Union[Mol, RWMol], enclosing_motif_id: List[Optional[int]]):
        self._mol = mol
        self._enclosing_motif_id = enclosing_motif_id

        self._atom_order = compute_canonical_atom_order(mol)
        self._node_id_to_position = {
            node_id: position for position, node_id in enumerate(self._atom_order)
        }

    def get_valid_start_node_choices(self) -> List[int]:
        """Return all valid node choices to start the generation with.

        Returns:
            List of ids of potential starting nodes.

        """
        # The default implementation considers starting anywhere.
        return list(range(self._mol.GetNumAtoms()))

    def pick_start_node(self, valid_first_node_choices: List[int]) -> int:
        """Picks a node to start the generation with.

        Args:
            valid_first_node_choices: Valid first node choices, exactly as returned from
                `get_valid_start_node_choices`.

        Returns:
            Id of the starting node.

        """
        # The default implementation is to pick a random valid choice; also see `pick_next_node`.
        return random.choice(valid_first_node_choices)

    @abstractmethod
    def get_valid_next_node_choices(self, frontier: List[int], visited: List[int]) -> List[int]:
        """Filters down the exploration queue to nodes that we want to consider for the next step.

        Args:
            frontier: Nodes that are one hop away from the current partial graph.
            visited: Nodes that are already present in the current partial graph.

        Returns:
            The list of ids of nodes from the exploration queue which are valid next nodes.

        """
        raise NotImplementedError()

    def pick_next_node(self, valid_next_node_choices: List[int]) -> int:
        """Picks the next node from the set of valid choices.

        Args:
            valid_next_node_choices: Valid next node choices, exactly as returned from
                `get_valid_next_node_choices`.

        Returns:
            The next node id, chosen from `valid_next_node_choices`.

        """
        # The default implementation is to choose randomly out of all valid choices (this supports
        # deterministic exploration orders which should use `len(valid_next_node_choices) == 1`).
        # It is possible to override this behaviour, although it's unclear if that's a good idea,
        # since choosing randomly from valid choices closely models the behaviour during inference.
        return random.choice(valid_next_node_choices)


class CanonicalOrder(GenerationOrder):
    def get_valid_start_node_choices(self) -> List[int]:
        # Deterministically pick the first node in the canonical order.
        return [self._atom_order[0]]

    def get_valid_next_node_choices(self, frontier: List[int], visited: List[int]) -> List[int]:
        # Pick the node that occurs first in the canonical order.
        return [min(frontier, key=self._node_id_to_position.get)]


class RandomOrder(GenerationOrder):
    def get_valid_next_node_choices(self, frontier: List[int], visited: List[int]) -> List[int]:
        # We consider all one-hop-away candidates as valid.
        return frontier


class LoopClosingOrder(GenerationOrder):
    def __init__(self, mol: Union[Mol, RWMol], enclosing_motif_id: List[Optional[int]]):
        super().__init__(mol, enclosing_motif_id)

        self._rings = [list(ring) for ring in GetSymmSSSR(mol)]

    def get_valid_next_node_choices(self, frontier: List[int], visited: List[int]) -> List[int]:
        # Keep track of rings with maximum overlap. Often there will be exactly one ring with
        # non-trivial overlap, but there may be several e.g. if three rings share a bond.
        max_overlap = 0
        max_overlap_rings = []

        for ring in self._rings:
            overlap = [node_id for node_id in ring if node_id in visited]
            n_overlap = len(overlap)

            # Check if the given ring is partially built.
            if 0 < n_overlap < len(ring):
                if n_overlap > max_overlap:
                    max_overlap = n_overlap
                    max_overlap_rings = [ring]
                elif n_overlap == max_overlap:
                    max_overlap_rings.append(ring)

        if max_overlap_rings:
            next_node_choices = [
                node_id
                for node_id in frontier
                if any(node_id in ring for ring in max_overlap_rings)
            ]
        else:
            next_node_choices = frontier

        return next_node_choices


class BFSOrder(GenerationOrder):
    def __init__(self, mol: Union[Mol, RWMol], enclosing_motif_id: List[Optional[int]]):
        super().__init__(mol, enclosing_motif_id)

        self._dist = {}
        self._last_dist = -1

    def get_valid_start_node_choices(self) -> List[int]:
        # Deterministically pick the first node in the canonical order.
        return [self._atom_order[0]]

    def get_valid_next_node_choices(self, frontier: List[int], visited: List[int]) -> List[int]:
        for node_id in frontier:
            if node_id not in self._dist:
                self._dist[node_id] = self._last_dist + 1

        # Sort by increasing distance from the start node.
        sorted_frontier = sorted(frontier, key=lambda node_id: self._dist[node_id])

        min_dist = self._dist[sorted_frontier[0]]
        max_dist = self._dist[sorted_frontier[-1]]

        # The frontier of BFS should contain nodes from at most two consecutive layers.
        assert max_dist - min_dist <= 1

        self._last_dist = min_dist
        return [node_id for node_id in frontier if self._dist[node_id] == min_dist]


class BFSOrderRandom(BFSOrder):
    def get_valid_start_node_choices(self) -> List[int]:
        return list(range(self._mol.GetNumAtoms()))
