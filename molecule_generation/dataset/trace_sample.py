"""Datatype for holding a graph generation trace sample."""
from typing import Dict, List, Optional, NamedTuple, Iterator

import numpy as np
from tf2_gnn import GraphSample


class TraceStep(NamedTuple):
    partial_node_features: np.ndarray
    partial_node_categorical_features: Optional[np.ndarray]
    partial_adjacency_lists: List[np.ndarray]
    correct_edge_choices: np.ndarray
    valid_edge_choices: np.ndarray
    edge_features: np.ndarray
    correct_edge_types: np.ndarray
    valid_edge_types: np.ndarray
    focus_node: int
    correct_attachment_point_choice: Optional[int]
    valid_attachment_point_choices: np.ndarray
    correct_node_type_choices: Optional[List[str]]


class TraceSample(GraphSample):
    def __init__(
        self,
        adjacency_lists: List[np.ndarray],
        type_to_node_to_num_inedges: np.ndarray,
        node_types: List[str],
        node_features: np.ndarray,
        node_categorical_features: Optional[np.ndarray],
        node_categorical_num_classes: Optional[int],
        partial_node_features: List[np.ndarray],
        partial_node_categorical_features: List[Optional[np.ndarray]],
        partial_adjacency_lists: List[List[np.ndarray]],
        correct_edge_choices: List[np.ndarray],
        valid_edge_choices: List[np.ndarray],
        correct_edge_types: List[np.ndarray],
        valid_edge_types: List[np.ndarray],
        focus_nodes: List[int],
        edge_features: List[np.ndarray],
        graph_property_values: Dict[str, float],
        correct_attachment_point_choices: Optional[List[Optional[int]]] = None,
        valid_attachment_point_choices: Optional[List[np.ndarray]] = None,
        correct_node_type_choices: Optional[List[Optional[List[str]]]] = None,
        correct_first_node_type_choices: Optional[List[str]] = None,
        mol=None,
    ):
        super().__init__(
            adjacency_lists=adjacency_lists,
            type_to_node_to_num_inedges=type_to_node_to_num_inedges,
            node_features=node_features,
        )
        self._node_types = node_types
        self._node_categorical_features = node_categorical_features
        self._node_categorical_num_classes = node_categorical_num_classes
        self._partial_node_features = partial_node_features
        self._partial_node_categorical_features = partial_node_categorical_features
        self._partial_adjacency_lists = partial_adjacency_lists
        self._correct_edge_choices = correct_edge_choices
        self._valid_edge_choices = valid_edge_choices
        self._correct_edge_types = correct_edge_types
        self._valid_edge_types = valid_edge_types
        self._focus_nodes = focus_nodes
        self._edge_features = edge_features
        self._graph_property_values = graph_property_values
        self._correct_attachment_point_choices = correct_attachment_point_choices
        self._valid_attachment_point_choices = valid_attachment_point_choices
        self._correct_node_type_choices = correct_node_type_choices
        self._correct_first_node_type_choices = correct_first_node_type_choices
        self._mol = mol

    def __iter__(self) -> Iterator[TraceStep]:
        for (
            partial_node_features,
            partial_node_categorical_features,
            partial_adjacency_lists,
            correct_edge_choices,
            valid_edge_choices,
            edge_features,
            correct_edge_types,
            valid_edge_types,
            focus_node,
            correct_attachment_point_choice,
            valid_attachment_point_choices,
            correct_node_type_choices,
        ) in zip(
            self.partial_node_features,
            self.partial_node_categorical_features,
            self.partial_adjacency_lists,
            self.correct_edge_choices,
            self.valid_edge_choices,
            self.edge_features,
            self.correct_edge_types,
            self.valid_edge_types,
            self.focus_nodes,
            self.correct_attachment_point_choices,
            self.valid_attachment_point_choices,
            self.correct_node_type_choices,
        ):
            yield TraceStep(
                partial_node_features=partial_node_features,
                partial_node_categorical_features=partial_node_categorical_features,
                partial_adjacency_lists=partial_adjacency_lists,
                correct_edge_choices=correct_edge_choices,
                valid_edge_choices=valid_edge_choices,
                edge_features=edge_features,
                correct_edge_types=correct_edge_types,
                valid_edge_types=valid_edge_types,
                focus_node=focus_node,
                correct_attachment_point_choice=correct_attachment_point_choice,
                valid_attachment_point_choices=valid_attachment_point_choices,
                correct_node_type_choices=correct_node_type_choices,
            )

    @property
    def node_types(self) -> List[str]:
        """String node types."""
        return self._node_types

    @property
    def node_features(self) -> np.ndarray:
        """Initial node features as ndarray of shape [V, ...]."""
        if isinstance(self._node_features, list):
            # Correct old trace samples that may contain List[List[float]] instead.
            self._node_features = np.array(self._node_features)

        return self._node_features

    @property
    def node_categorical_features(self) -> Optional[np.ndarray]:
        """Additional node categorical features as ndarray of shape [V]."""
        return getattr(self, "_node_categorical_features", None)

    @property
    def node_categorical_num_classes(self) -> Optional[int]:
        """Number of classes for node categorical features."""
        return getattr(self, "_node_categorical_num_classes", None)

    @property
    def partial_node_features(self) -> List[np.ndarray]:
        """The node features for each partial graph."""
        return self._partial_node_features

    @property
    def partial_node_categorical_features(self) -> List[Optional[np.ndarray]]:
        """The categorical node features for each partial graph."""
        return getattr(self, "_partial_node_categorical_features", [None] * len(self.focus_nodes))

    @property
    def partial_adjacency_lists(self) -> List[List[np.ndarray]]:
        """Partial adjacency list of a single point in the generation trace."""
        return self._partial_adjacency_lists

    @property
    def correct_edge_choices(self) -> List[List[np.ndarray]]:
        """Correct edge choices of a single point in the generation trace."""
        return self._correct_edge_choices

    @property
    def valid_edge_choices(self) -> List[List[np.ndarray]]:
        return self._valid_edge_choices

    @property
    def correct_edge_types(self) -> List[List[np.ndarray]]:
        return self._correct_edge_types

    @property
    def valid_edge_types(self) -> List[List[np.ndarray]]:
        return self._valid_edge_types

    @property
    def correct_attachment_point_choices(self) -> List[Optional[int]]:
        value = getattr(self, "_correct_attachment_point_choices", None)
        return value or [None] * len(self.focus_nodes)

    @property
    def valid_attachment_point_choices(self) -> List[np.ndarray]:
        value = getattr(self, "_valid_attachment_point_choices", None)
        return value or [
            np.zeros(
                0,
            )
            for _ in range(len(self.focus_nodes))
        ]

    @property
    def focus_nodes(self) -> List[int]:
        return self._focus_nodes

    @property
    def edge_features(self) -> List[np.ndarray]:
        return self._edge_features

    @property
    def graph_property_values(self) -> Dict[str, np.ndarray]:
        return self._graph_property_values

    @property
    def correct_node_type_choices(self) -> List[Optional[List[str]]]:
        value = getattr(self, "_correct_node_type_choices", None)
        return value or [None] * len(self.focus_nodes)

    @property
    def correct_first_node_type_choices(self) -> List[str]:
        value = getattr(self, "_correct_first_node_type_choices", None)
        return value or ["C"]

    @property
    def mol(self):
        return self._mol

    def __str__(self):
        return (
            f"Adjacency lists: \n{self.adjacency_lists}\n"
            f"Node types: \n{self.node_types}"
            f"Node features: \n{self.node_features}\n"
            f"Partial adjacency lists: \n{self.partial_adjacency_lists}\n"
            f"Correct edge choices: \n{self.correct_edge_choices}\n"
            f"Valid edge choices: \n{self.valid_edge_choices}\n"
            f"Correct edge types: \n{self.correct_edge_types}\n"
            f"Valid edge types: \n{self.valid_edge_types}\n"
            f"Focus nodes: \n{self.focus_nodes}\n"
            f"Edge features: \n{self.edge_features}\n"
            f"Graph properties: \n{self._graph_property_values}\n"
        )
