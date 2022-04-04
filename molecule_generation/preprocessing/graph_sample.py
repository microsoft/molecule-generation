"""Graph sample datatype."""
from typing import NamedTuple, List, Dict, Union, Optional
from rdkit.Chem import Mol, RWMol

from molecule_generation.chem.motif_utils import MotifAnnotation


NodeTypes = List[str]


class Edge(NamedTuple):
    source: int
    target: int
    type: int


AdjacencyList = List[Edge]
GraphProperties = Dict[str, float]


class GraphSample(NamedTuple):
    adjacency_list: AdjacencyList
    num_edge_types: int
    node_features: List[List[float]]
    graph_properties: GraphProperties
    node_types: NodeTypes
    smiles_string: str
    motifs: List[MotifAnnotation] = []
    node_categorical_features: Optional[List[int]] = None
    node_categorical_num_classes: Optional[int] = None
    mol: Optional[Union[Mol, RWMol]] = None


class GraphTraceStep(NamedTuple):
    partial_graph: GraphSample
    focus_node: int
    correct_edge_choices: List[Edge]
    valid_edge_choices: List[Edge]
    distance_to_target: List[int]
    correct_attachment_point_choice: Optional[int]
    valid_attachment_point_choices: List[int]
    correct_node_type_choices: NodeTypes


class GraphTraceSample(NamedTuple):
    full_graph: GraphSample
    partial_graphs: List[GraphSample]
    focus_nodes: List[int]
    correct_edge_choices: List[List[Edge]]
    valid_edge_choices: List[List[Edge]]
    distance_to_target: List[List[int]]
    correct_attachment_point_choices: List[Optional[int]]
    valid_attachment_point_choices: List[List[int]]
    correct_node_type_choices: List[Optional[NodeTypes]]
    correct_first_node_type_choices: NodeTypes

    def __iter__(self) -> GraphTraceStep:
        for (
            partial_graph,
            focus_node,
            correct_edge_choice,
            valid_edge_choice,
            distance,
            correct_attachment_point_choice,
            valid_attachment_point_choices,
            correct_node_type_choice,
        ) in zip(
            self.partial_graphs,
            self.focus_nodes,
            self.correct_edge_choices,
            self.valid_edge_choices,
            self.distance_to_target,
            self.correct_attachment_point_choices,
            self.valid_attachment_point_choices,
            self.correct_node_type_choices,
        ):
            yield GraphTraceStep(
                partial_graph=partial_graph,
                focus_node=focus_node,
                correct_edge_choices=correct_edge_choice,
                valid_edge_choices=valid_edge_choice,
                distance_to_target=distance,
                correct_attachment_point_choice=correct_attachment_point_choice,
                valid_attachment_point_choices=valid_attachment_point_choices,
                correct_node_type_choices=correct_node_type_choice,
            )
