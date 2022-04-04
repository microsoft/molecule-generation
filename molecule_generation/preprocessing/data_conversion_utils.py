"""Utility functions for various data conversions."""
import logging
from functools import partial
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Type, Union

import numpy as np
from rdkit.Chem import GetMolFrags
from rdkit.Chem.rdchem import Atom, BondType, Mol, RWMol
from tf2_gnn.data.utils import get_tied_edge_types, process_adjacency_lists

from molecule_generation.chem.rdkit_helpers import remove_atoms_outside_frag
from molecule_generation.preprocessing.cgvae_generation_trace import graph_sample_to_cgvae_trace
from molecule_generation.preprocessing.generation_order import GenerationOrder, BFSOrder
from molecule_generation.preprocessing.moler_generation_trace import graph_sample_to_MoLeR_trace
from molecule_generation.preprocessing.graph_sample import Edge, GraphSample, GraphTraceStep
from molecule_generation.dataset.trace_sample import TraceSample
from molecule_generation.chem.motif_utils import (
    MotifAnnotation,
    MotifAtomAnnotation,
    MotifVocabulary,
)
from molecule_generation.utils.sequential_worker_pool import get_worker_pool
from molecule_generation.chem.topology_features import calculate_topology_features
from molecule_generation.chem.valence_constraints import (
    constrain_edge_choices_based_on_valence,
    constrain_edge_types_based_on_valence,
)
from molecule_generation.chem.atom_feature_utils import AtomFeatureExtractor
from molecule_generation.chem.molecule_dataset_utils import featurise_atoms

logger = logging.getLogger(__name__)


def onehot_encoding(labels: np.ndarray, num_classes: int):
    """
    Onehot encoding of integer labels.

    Args:
        labels: a 1-dimensional array of integer labels.
        num_classes: the number of classes. Should be at least max(labels) + 1.

    Returns:
        A onehot encoding of labels.

    >>> labels = np.array([0, 1, 0, 2])
    >>> onehot_encoding(labels, 3)
    array([[1, 0, 0],
           [0, 1, 0],
           [1, 0, 0],
           [0, 0, 1]], dtype=int32)

    """
    return np.eye(num_classes, dtype=np.int32)[labels]


def numpyify_edge_choices_and_types(edge_choices: List[Edge]) -> np.ndarray:
    """Convert a list of edges into a numpy array (adjacency list)

    Args:
        edge_choices: A list of Edge objects.

    Returns:
        A numpy array of shape (len(edge_choices), 2). The ith row of the array corresponds to the
        ith edge in the edge_choices list. Column 0 contains the edge sources, and column 1 contains
        the edge targets.

    >>> edge_choices = [Edge(0, 3, 4), Edge(0, 1, 2), Edge(3, 2, 5)]
    >>> numpyify_edge_choices_and_types(edge_choices)
    (array([[0, 3],
           [0, 1],
           [3, 2]], dtype=int32), array([4, 2, 5], dtype=int32))
    """
    edges = np.zeros(shape=(len(edge_choices), 2), dtype=np.int32)
    edge_types = np.zeros(shape=(len(edge_choices)), dtype=np.int32)
    for i, edge in enumerate(edge_choices):
        edges[i, 0] = edge.source
        edges[i, 1] = edge.target
        edge_types[i] = edge.type
    return edges, edge_types


def convert_graph_sample_to_adjacency_list(
    graph_sample: GraphSample,
    num_forward_edge_types: int,
    tie_fwd_bkwd_edges: Union[bool, List[int]],
    add_self_loops: bool,
) -> List[np.ndarray]:
    """Extract a set of adjacency lists from a graph sample.

    Args:
        graph_sample: the GraphSample object to be converted.
        num_forward_edge_types: The number of 'forward' edge types. This should be equal to the
            number of distinct edge types in the graph, ignoring self loops and direction. In the
            case of molecules for example, these would single, double and triple bonds, and so this
            number would be 3.
        tie_fwd_bkwd_edges: One of
                - a set of edge types that should be treated as undirected
                - `True` if all edge types are undirected
                - `False` if all edge types are directed
        add_self_loops: True if we want to add self loops to the resulting adjacency list. If we do,
            the self loop edge type will be stored in the last index of the adjacency lists.

    >>> graph_sample = GraphSample(
    ...     adjacency_list=[
    ...         Edge(source=0, target=1, type=0),
    ...         Edge(source=1, target=2, type=2),
    ...         Edge(source=2, target=3, type=0),
    ...         Edge(source=3, target=4, type=1),
    ...     ],
    ...     num_edge_types=3,
    ...     node_features=[0.0 for _ in range(5)],
    ...     graph_properties={"sa_score": 8.9},
    ...     node_types=["C" for _ in range(5)],
    ...     smiles_string="not really a SMILES string",
    ... )
    >>> convert_graph_sample_to_adjacency_list(graph_sample, 3, True, False)
    [array([[0, 1],
           [2, 3],
           [1, 0],
           [3, 2]], dtype=int32), array([[3, 4],
           [4, 3]], dtype=int32), array([[1, 2],
           [2, 1]], dtype=int32)]
    """
    adjacency_lists = [[] for _ in range(num_forward_edge_types)]

    for edge in graph_sample.adjacency_list:
        adjacency_lists[edge.type].append((edge.source, edge.target))

    # If there are no edges, we assume a single-node graph (this does matter for self-loops).
    max_node_idx = max(
        sum([[edge.source, edge.target] for edge in graph_sample.adjacency_list], [0])
    )

    tied_fwd_bkwd_edge_types = get_tied_edge_types(
        tie_fwd_bkwd_edges=tie_fwd_bkwd_edges, num_fwd_edge_types=num_forward_edge_types
    )

    adjacency_lists, _ = process_adjacency_lists(
        adjacency_lists=adjacency_lists,
        num_nodes=max_node_idx + 1,
        add_self_loop_edges=add_self_loops,
        tied_fwd_bkwd_edge_types=tied_fwd_bkwd_edge_types,
        self_loop_edge_type=-1,
    )

    return adjacency_lists


def correct_edge_to_multihot_encoding(correct_edges: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Convert a correct index list into a multi-hot encoding of the valid edge choices.

    Args:
        correct_edges: a numpy array of shape (NC, 2) of ints. The first column is the source
            index of each edge, the second column is the target index.
        edges: a numpy array of shape (N, 2) in ints. The first column is the source
            index of each edge, the second column is the target index.

    Returns:
        A flot32 array of 0s and 1s of shape (N,). It will be 1 at the indices where `edges` has a
        target node that is one of the target nodes of `correct_edges`.

    >>> edges = np.array([[0, 0], [0, 1], [0, 3], [0, 4], [0, 5], [0, 8]])
    >>> correct_edges = np.array([[0, 3], [0, 5]])
    >>> correct_edge_to_multihot_encoding(correct_edges, edges)
    array([0., 0., 1., 0., 1., 0.], dtype=float32)

    """
    correct_targets = correct_edges[:, 1]
    valid_targets = edges[:, 1]
    multihot = np.isin(valid_targets, correct_targets).astype(np.float32)
    if not np.sum(multihot) == len(correct_targets):
        raise ValueError(
            f"Warning! Something has gone wrong in the correct_edge_to_multihot_encoding_function."
        )
    return multihot


def convert_graph_samples_to_traces(
    graph_sample_data: Iterable[GraphSample],
    num_fwd_edge_types: int,
    tie_fwd_bkwd_edges: bool,
    add_self_loop_edges: bool,
    atom_feature_extractors: List[AtomFeatureExtractor],
    motif_vocabulary: Optional[MotifVocabulary] = None,
    num_processes: int = 8,
    chunksize: int = 10,
    MoLeR_style_trace: bool = False,
    generation_order_cls: Type[GenerationOrder] = BFSOrder,
) -> Iterator[TraceSample]:
    """
    Convert an iterable of GraphSample objects into an iterable of GraphTrace objects.

    Args:
        graph_sample_data: An iterable of GraphSample objects which we want to convert.
        num_fwd_edge_types: The number of edge types, not including self loops and discounting the
            direction of the edges. For example, if the dataset of molecules contains only single,
            double and triple bonds, then this parameter would be 3.
        tie_fwd_bkwd_edges: True if we treat edges in both directions as the same type, False
            otherwise. It should be true if we interpret the GraphSample objects as undirected
            graphs.
        add_self_loop_edges: True if we add self loop edges to the resulting TraceSample, False
            otherwise.
        num_processes: The number of parallel processes to use to perform the mapping.
        chunksize: The chuncksize to pass to each parallel process. See the
            multiprocessing.Pool.imap documentation for details.
        MoLeR_style_trace: whether the function should convert generation traces using the
            new molecule level representation trace behaviour or the (original) atom level
            representation behaviour.

    Returns:
        An iterator of TraceSample objects.

    """

    convert = partial(
        _convert_graph_sample_to_graph_trace,
        num_fwd_edge_types=num_fwd_edge_types,
        tie_fwd_bkwd_edges=tie_fwd_bkwd_edges,
        add_self_loop_edges=add_self_loop_edges,
        atom_feature_extractors=atom_feature_extractors,
        motif_vocabulary=motif_vocabulary,
        MoLeR_style_trace=MoLeR_style_trace,
        generation_order_cls=generation_order_cls,
    )
    logger.debug(
        f"Converting to graph traces with {num_processes} processes and chunksize {chunksize}."
    )
    with get_worker_pool(num_processes) as p:
        for maybe_converted in p.imap(convert, graph_sample_data, chunksize):
            if isinstance(maybe_converted, tuple):
                logger.warning(
                    f"Unable to convert {maybe_converted[0]}."
                    f"Error message was: " + str(maybe_converted[1])
                )
            else:
                yield maybe_converted


def _convert_graph_sample_to_graph_trace(
    graph: GraphSample,
    num_fwd_edge_types: int,
    tie_fwd_bkwd_edges: bool,
    add_self_loop_edges: bool,
    atom_feature_extractors: List[AtomFeatureExtractor],
    motif_vocabulary: Optional[MotifVocabulary] = None,
    include_generation_trace: bool = True,
    MoLeR_style_trace: bool = False,
    generation_order_cls: Type[GenerationOrder] = BFSOrder,
    save_molecule_to_trace: bool = False,
) -> Union[Tuple, TraceSample]:
    """Convert a single GraphSample to a TraceSample object.

    Returns a tuple of the original graph sample and the resulting error if it was impossible to
    convert the graph to a generation trace.

    See convert_graph_samples_to_graph_traces for parameter details.
    """
    try:
        return __inner_convert_graph_sample_to_graph_trace(
            graph=graph,
            num_fwd_edge_types=num_fwd_edge_types,
            tie_fwd_bkwd_edges=tie_fwd_bkwd_edges,
            add_self_loop_edges=add_self_loop_edges,
            atom_feature_extractors=atom_feature_extractors,
            motif_vocabulary=motif_vocabulary,
            include_generation_trace=include_generation_trace,
            MoLeR_style_trace=MoLeR_style_trace,
            generation_order_cls=generation_order_cls,
            save_molecule_to_trace=save_molecule_to_trace,
        )
    except Exception as e:
        print(f"Converting graph sample for {graph.smiles_string} failed - aborting!")
        return graph, e


def __inner_convert_graph_sample_to_graph_trace(
    graph: GraphSample,
    num_fwd_edge_types: int,
    tie_fwd_bkwd_edges: bool,
    add_self_loop_edges: bool,
    atom_feature_extractors: List[AtomFeatureExtractor],
    motif_vocabulary: Optional[MotifVocabulary] = None,
    include_generation_trace: bool = True,
    MoLeR_style_trace: bool = False,
    generation_order_cls: Type[GenerationOrder] = BFSOrder,
    save_molecule_to_trace: bool = False,
) -> Union[Tuple, TraceSample]:
    try:
        if include_generation_trace:
            if MoLeR_style_trace:
                generation_trace = graph_sample_to_MoLeR_trace(graph, generation_order_cls)
                # Reset graph to what is used in the generation trace, which may have changed the node order:
                graph = generation_trace.full_graph
            else:
                assert generation_order_cls is BFSOrder, "For CGVAE only BFS order is supported."
                generation_trace = graph_sample_to_cgvae_trace(graph)

            correct_first_node_type_choices = generation_trace.correct_first_node_type_choices
        else:
            generation_trace = []
            correct_first_node_type_choices = None
    except ValueError as e:
        return graph, e
    full_adjacency_list = convert_graph_sample_to_adjacency_list(
        graph, num_fwd_edge_types, tie_fwd_bkwd_edges, add_self_loop_edges
    )
    partial_adjacency_lists = []
    correct_edge_choices = []
    valid_edge_choices = []
    correct_edge_types = []
    valid_edge_types = []
    correct_attachment_point_choices = []
    valid_attachment_point_choices = []
    edge_feature_list = []
    partial_node_feature_list = []
    partial_node_categorical_feature_list = []
    focus_node_list = []
    correct_node_type_choices = []
    for trace_step in generation_trace:
        trace_step: GraphTraceStep

        valid_edges, _ = numpyify_edge_choices_and_types(trace_step.valid_edge_choices)
        partial_adjacency_list = convert_graph_sample_to_adjacency_list(
            trace_step.partial_graph,
            num_fwd_edge_types,
            tie_fwd_bkwd_edges,
            add_self_loop_edges,
        )

        correct_attachment_point_choices.append(trace_step.correct_attachment_point_choice)
        valid_attachment_point_choices.append(np.array(trace_step.valid_attachment_point_choices))
        correct_node_type_choices.append(trace_step.correct_node_type_choices)

        edge_constraint_mask = constrain_edge_choices_based_on_valence(
            start_node=trace_step.focus_node,
            candidate_target_nodes=valid_edges[:, 1],
            adjacency_lists=partial_adjacency_list,
            node_types=graph.node_types,
        )
        constrained_edges = valid_edges[edge_constraint_mask]

        if not MoLeR_style_trace and constrained_edges.size == 0:
            continue

        # The next line is needed to ensure that all properties of the atoms in the molecule
        # (valences etc.) are calculated before the feature extractors are called.
        try:
            trace_step.partial_graph.mol.UpdatePropertyCache()
        except Exception:
            continue

        focus_node_list.append(trace_step.focus_node)
        partial_adjacency_lists.append(partial_adjacency_list)
        correct_edges, edge_types = numpyify_edge_choices_and_types(trace_step.correct_edge_choices)
        onehot_edge_types = onehot_encoding(edge_types, graph.num_edge_types)
        edge_type_mask = constrain_edge_types_based_on_valence(
            start_node=trace_step.focus_node,
            candidate_target_nodes=correct_edges[:, 1],
            adjacency_lists=partial_adjacency_list,
            node_types=graph.node_types,
        )
        # Make sure we have not made a mistake anywhere!
        if not (edge_type_mask >= onehot_edge_types).all():
            print(
                f"\n\nWarning: edge type mask error for molecule:\n{graph.smiles_string}"
                f"\tHave type validity mask {edge_type_mask} and labels {onehot_edge_types}"
            )
            edge_type_mask = np.maximum(edge_type_mask, onehot_edge_types)

        correct_edge_types.append(onehot_edge_types)
        valid_edge_types.append(edge_type_mask)

        try:
            multihot_correct_edges = correct_edge_to_multihot_encoding(
                correct_edges, constrained_edges
            )
        except ValueError as e:
            return graph, e

        # Calculate the features of nodes in the partial graph:
        features = featurise_atoms(
            mol=trace_step.partial_graph.mol,
            atom_feature_extractors=atom_feature_extractors,
            motif_vocabulary=motif_vocabulary,
            motifs=graph.motifs,
        )

        partial_node_features = np.concatenate([features.real_valued_features])
        partial_node_feature_list.append(partial_node_features)

        if features.categorical_features is not None:
            partial_node_categorical_features = np.array(features.categorical_features)
        else:
            partial_node_categorical_features = None

        partial_node_categorical_feature_list.append(partial_node_categorical_features)

        # Calculate the features of potential edges:
        constrained_distances = np.array(trace_step.distance_to_target, dtype=np.float32)[
            edge_constraint_mask
        ].reshape(-1, 1)
        topology_features = calculate_topology_features(
            constrained_edges, trace_step.partial_graph.mol
        )
        edge_features = np.concatenate([constrained_distances, topology_features], axis=-1)
        edge_feature_list.append(edge_features)

        correct_edge_choices.append(multihot_correct_edges)
        valid_edge_choices.append(constrained_edges)

    if graph.node_categorical_features is not None:
        node_categorical_features = np.array(graph.node_categorical_features)
    else:
        node_categorical_features = None

    return TraceSample(
        adjacency_lists=full_adjacency_list,
        type_to_node_to_num_inedges=None,  # This is not used anywhere at the moment.
        node_types=graph.node_types,
        node_features=np.array(graph.node_features),
        node_categorical_features=node_categorical_features,
        node_categorical_num_classes=graph.node_categorical_num_classes,
        partial_adjacency_lists=partial_adjacency_lists,
        correct_edge_choices=correct_edge_choices,
        valid_edge_choices=valid_edge_choices,
        correct_edge_types=correct_edge_types,
        valid_edge_types=valid_edge_types,
        focus_nodes=focus_node_list,
        edge_features=edge_feature_list,
        partial_node_features=partial_node_feature_list,
        partial_node_categorical_features=partial_node_categorical_feature_list,
        graph_property_values=graph.graph_properties,
        correct_attachment_point_choices=correct_attachment_point_choices,
        valid_attachment_point_choices=valid_attachment_point_choices,
        correct_node_type_choices=correct_node_type_choices,
        correct_first_node_type_choices=correct_first_node_type_choices,
        mol=graph.mol if save_molecule_to_trace else None,
    )


def convert_jsonl_file_to_graph_samples(
    data: List[Dict[str, Any]], num_processes: int = 8, chunksize: int = 10
) -> Iterator[GraphSample]:
    """Convert the result of reading a jsonl file to a list of GraphSample objects.

    Each element in the list must be a dictionary with key "graph", whose value is a dictionary
    with keys "node_labels", "node_features" and "adjacency_lists"
    """
    logger.debug(
        f"Converting to graph samples with {num_processes} processes and chunksize {chunksize}."
    )
    with get_worker_pool(num_processes) as p:
        yield from p.imap(_convert_single_jsonl_to_graph_sample, data, chunksize=chunksize)


def _convert_single_jsonl_to_graph_sample(datum: Dict[str, Any]) -> GraphSample:
    """Convert from a standard datum loaded from the jsonl format to a GraphSample object."""
    assert "graph" in datum.keys(), "The datum must contain graph information in the 'graph' key."
    graph = datum["graph"]
    assert {"node_types", "node_features", "adjacency_lists"}.issubset(graph.keys()), (
        f"The graph representation must contain keys 'node_features', 'node_types', and "
        f"'adjacency_lists' in its keys. It contains {graph.keys()}"
    )

    adjacency_lists = graph["adjacency_lists"]
    num_edge_types = len(adjacency_lists)
    converted_adjacency_list = []
    for i, adjacency_list in enumerate(adjacency_lists):
        if not adjacency_list:
            continue
        for edge in adjacency_list:
            converted_adjacency_list.append(Edge(source=edge[0], target=edge[1], type=i))

    motifs = []

    for motif_type, atom_annotations in datum.get("motifs", []):
        motifs.append(
            MotifAnnotation(
                motif_type=motif_type,
                atoms=[
                    MotifAtomAnnotation(atom_id, symmetry_class_id)
                    for atom_id, symmetry_class_id in atom_annotations
                ],
            )
        )

    return GraphSample(
        adjacency_list=converted_adjacency_list,
        num_edge_types=num_edge_types,
        node_features=graph["node_features"],
        node_categorical_features=graph.get("node_categorical_features"),
        node_categorical_num_classes=graph.get("node_categorical_num_classes"),
        graph_properties=datum.get("properties", {}),
        node_types=graph["node_types"],
        smiles_string=datum["SMILES"],
        motifs=motifs,
    )


def convert_adjacency_list_to_romol(
    atom_types: List[str],
    adjacency_lists: List[np.ndarray],
    adjacency_list_to_bond_type: Dict[int, BondType],
) -> Mol:
    """Convert nodes and adjacency lists to a ROMol.

    Args:
        atom_types: a list of string representations of the atom types in the molecule.
        adjacency_lists: a list of numpy arrays, each of which should have shape (E, 2). Each list
            can represent a different bond type.
        adjacency_list_to_bond_type: a dictionary that maps the index of the adjacency list in the
            above list to the bond type that is represented by that list.

    Returns:
        A ROMol that has the atom types given by node_types and the bonds specified by the given
        adjacency lists.

    Constructing a (Kekulized) Benzine ring:
    >>> atom_types = ["C"] * 6
    >>> adjacency_lists = [np.array([[0, 1], [2, 3], [4, 5]]),
    ...                    np.array([[1, 2], [3, 4], [5, 0]])]
    >>> adj_list_to_bond_type = {0: BondType.SINGLE, 1: BondType.DOUBLE}
    >>> mol = convert_adjacency_list_to_romol(atom_types, adjacency_lists, adj_list_to_bond_type)
    >>> from rdkit.Chem import MolToSmiles
    >>> MolToSmiles(mol)
    'C1=CC=CC=C1'
    """
    mol = RWMol()
    for atom_symbol in atom_types:
        mol.AddAtom(Atom(atom_symbol))
    for adj_list_idx, adjacency_list in enumerate(adjacency_lists):
        bond_type = adjacency_list_to_bond_type[adj_list_idx]
        for bond in adjacency_list:
            mol.AddBond(int(bond[0]), int(bond[1]), bond_type)
    mol = remove_non_max_frags(mol)
    return Mol(mol)


def remove_non_max_frags(mol: RWMol) -> RWMol:
    """Remove all but the biggest connected component."""
    frags: Tuple[Tuple[int, ...], ...] = GetMolFrags(mol)
    # Easy out if there is only one fragment.
    if len(frags) == 1:
        return mol

    largest_frag_idx = np.argmax([len(frag) for frag in frags])
    # largest_frag is a set of the indices of the atoms in the largest connected component of the
    # given molecule.
    largest_frag = set(frags[largest_frag_idx])

    return remove_atoms_outside_frag(mol, largest_frag)


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
