"""A wrapper a general generation trace dataset."""
import logging
import random
from abc import abstractmethod
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import tensorflow as tf
from tf2_gnn import DataFold, GraphBatchTFDataDescription, GraphDataset

from molecule_generation.dataset.trace_sample import TraceSample, TraceStep
from molecule_generation.utils.data_utils import safe_concat, safe_make_array


logger = logging.getLogger(__name__)


class TraceDataset(GraphDataset[TraceSample]):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        super_params = super().get_default_hyperparameters()
        these_hypers: Dict[str, Any] = {
            "max_nodes_per_batch": 10000,
            "max_partial_nodes_per_batch": 30000,
            "trace_element_keep_prob": 0.5,
            "add_self_loop_edges": True,
            "tie_fwd_bkwd_edges": True,
            # A list of graph properties to expose, if present in the underlying data.
            # For convenience, we allow model configuration here as well (so that it can
            # live in a single place and configuration doesn't require changing two settings
            # in lockstep):
            "graph_properties": {
                "sa_score": {
                    "type": "regression",
                    "loss_weight_factor": 1.0,
                    "normalise_loss": False,
                },
            },
        }
        super_params.update(these_hypers)
        return super_params

    def __init__(self, params: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        logger.info("Initialising TraceDataset.")
        super().__init__(params, metadata=metadata)

        self._num_edge_types_to_classify = 3
        if params["tie_fwd_bkwd_edges"]:
            self._num_edge_types = self._num_edge_types_to_classify
        else:
            self._num_edge_types = 2 * self._num_edge_types_to_classify
        self._num_edge_types += int(params["add_self_loop_edges"])

        # Prepare data for the graph property prediction:
        self._graph_property_names = list(self._params.get("graph_properties", {}).keys())

        # Prepare fields that we will lazily fill:
        self._node_feature_shape = None
        self._partial_node_feature_shape = None
        self._num_node_types = None
        self._loaded_feature_shapes = False

        self._loaded_data: Dict[DataFold, List[TraceSample]] = {}
        logger.debug("Done initialising TraceDataset.")

    def _new_batch(self) -> Dict[str, Any]:
        batch = {
            "node_features": [],
            "node_categorical_features": [],
            # More adjacency lists are appended in the `_finalise_batch` method, depending on the
            # parameters `tie_fwd_bkwd_edges` and `add_self_loop_edges`.
            "adjacency_lists": [[] for _ in range(self.num_edge_types_to_classify)],
            "node_to_graph_map": [],
            "num_graphs_in_batch": 0,
            "num_nodes_in_batch": 0,
            "graph_property_to_values": {
                property_name: [] for property_name in self._graph_property_names
            },
            "graph_property_to_graph_ids": {
                property_name: [] for property_name in self._graph_property_names
            },
            # --- Data holders for the traces ---
            "node_types": [],
            "node_to_partial_graph_map": [],
            "focus_nodes": [],
            "partial_node_features": [],
            "partial_node_categorical_features": [],
            # More adjacency lists are appended in the `_finalise_batch` method, depending on the
            # parameters `tie_fwd_bkwd_edges` and `add_self_loop_edges`.
            "partial_adjacency_lists": [[] for _ in range(self.num_edge_types_to_classify)],
            "correct_edge_choices": [],
            "num_correct_edge_choices": [],
            "stop_node_label": [],
            "valid_edge_choices": [],
            "num_valid_attachment_point_choices_in_batch": 0,
            "correct_attachment_point_choices": [],
            "valid_attachment_point_choices": [],
            "edge_features": [],
            "correct_edge_types": [],
            "valid_edge_types": [],
            "partial_node_to_original_node_map": [],
            "partial_graph_to_original_graph_map": [],
            "num_partial_graphs_in_batch": 0,
            "num_partial_nodes_in_batch": 0,
            "partial_graphs_requiring_node_choices": [],
            "correct_node_type_choices": [],
            "correct_first_node_type_choices": [],
        }
        return batch

    @property
    def num_edge_types_to_classify(self):
        """The number of 'true' edge types. Usually 3, for single, double and triple bonds."""
        return self._num_edge_types_to_classify

    def _batch_would_be_too_full(
        self, raw_batch: Dict[str, Any], graph_sample: TraceSample
    ) -> bool:
        """Return whether the current raw batch would be too full if graph_sample was added."""
        num_nodes_in_graph = len(graph_sample.node_features)
        nodes_full = (
            raw_batch["num_nodes_in_batch"] + num_nodes_in_graph
            > self._params["max_nodes_per_batch"]
        )
        partial_nodes_full = (
            raw_batch["num_partial_nodes_in_batch"] + num_nodes_in_graph
            > self._params["max_partial_nodes_per_batch"]
        )
        return nodes_full or partial_nodes_full

    def _include_trace_step_in_batch(self, trace_step: TraceStep) -> bool:
        return random.uniform(0, 1) < self._params["trace_element_keep_prob"]

    def _add_graph_to_batch(self, raw_batch: Dict[str, Any], graph_sample: TraceSample) -> None:
        """Add the full graph and partial graphs to the batch.

        A similar batching strategy to that for the full graphs is implemented here for the partial
        graphs. The edge connections are stored in the "partial_adjacency_lists" value of the raw
        batch.

        The mapping from partial graph to full graph is stored in the
        "partial_node_to_original_node_map". As an example, if there is an edge (u, v) in the
        partial adjacency lists, this corresponds to an edge (u', v') in the original graph, where
        u' = partial_node_to_original_node_map[u], and similar for v'.
        """
        super()._add_graph_to_batch(raw_batch, graph_sample)
        original_graph_id_in_batch = raw_batch["num_graphs_in_batch"]

        if self.uses_categorical_features:
            raw_batch["node_categorical_features"].extend(graph_sample.node_categorical_features)

        raw_batch["node_types"].extend(self.node_types_to_indices(graph_sample.node_types))

        for trace_step in graph_sample:
            partial_graph_id_in_batch = raw_batch["num_partial_graphs_in_batch"]
            num_nodes_in_partial_graph = len(trace_step.partial_node_features)
            # Only add new partial graphs if that will not make the batch too big.
            if self._batch_would_be_too_full(raw_batch, graph_sample):
                break

            # Allow subsampling of generation trace steps:
            if not self._include_trace_step_in_batch(trace_step):
                continue

            # Actually start adding partial graphs to the batch.
            raw_batch["node_to_partial_graph_map"].append(
                np.full(
                    shape=[num_nodes_in_partial_graph],
                    fill_value=partial_graph_id_in_batch,
                    dtype=np.int32,
                )
            )
            raw_batch["focus_nodes"].append(
                trace_step.focus_node + raw_batch["num_partial_nodes_in_batch"]
            )
            raw_batch["edge_features"].append(trace_step.edge_features)
            raw_batch["partial_node_features"].append(trace_step.partial_node_features)
            for edge_type_idx, batch_partial_adjacency_list in enumerate(
                raw_batch["partial_adjacency_lists"]
            ):
                batch_partial_adjacency_list.append(
                    trace_step.partial_adjacency_lists[edge_type_idx].reshape(-1, 2)
                    + raw_batch["num_partial_nodes_in_batch"]
                )
            raw_batch["correct_edge_choices"].append(trace_step.correct_edge_choices)
            num_correct_edge_choices = np.sum(trace_step.correct_edge_choices)
            raw_batch["num_correct_edge_choices"].append(num_correct_edge_choices)
            raw_batch["stop_node_label"].append(int(num_correct_edge_choices == 0))
            raw_batch["valid_edge_choices"].append(
                trace_step.valid_edge_choices + raw_batch["num_partial_nodes_in_batch"]
            )
            raw_batch["correct_edge_types"].append(trace_step.correct_edge_types)
            raw_batch["valid_edge_types"].append(trace_step.valid_edge_types)

            if self.uses_categorical_features:
                raw_batch["partial_node_categorical_features"].extend(
                    trace_step.partial_node_categorical_features
                )

            if trace_step.correct_attachment_point_choice is not None:
                raw_batch["correct_attachment_point_choices"].append(
                    list(trace_step.valid_attachment_point_choices).index(
                        trace_step.correct_attachment_point_choice
                    )
                    + raw_batch["num_valid_attachment_point_choices_in_batch"]
                )

                raw_batch["valid_attachment_point_choices"].append(
                    trace_step.valid_attachment_point_choices
                    + raw_batch["num_partial_nodes_in_batch"]
                )

                raw_batch["num_valid_attachment_point_choices_in_batch"] += len(
                    trace_step.valid_attachment_point_choices
                )

            # Note: This is only correct for the JSONLTraceDataset case, as we are remapping
            # in the JSONLMoLeRTraceDataset case. In the JSONLMoLeRTraceDataset, this is a partial
            # node to reindexed node map.
            raw_batch["partial_node_to_original_node_map"].append(
                np.arange(num_nodes_in_partial_graph) + raw_batch["num_nodes_in_batch"]
            )
            raw_batch["partial_graph_to_original_graph_map"].append(original_graph_id_in_batch)

            # And finally, the correct node type choices. Here, we have an empty list of
            # correct choices for all steps where we didn't choose a node, so we skip that:
            if trace_step.correct_node_type_choices is not None:
                raw_batch["partial_graphs_requiring_node_choices"].append(partial_graph_id_in_batch)
                raw_batch["correct_node_type_choices"].append(
                    self.node_types_to_multi_hot(trace_step.correct_node_type_choices)
                )

            raw_batch["num_partial_graphs_in_batch"] += 1
            raw_batch["num_partial_nodes_in_batch"] += num_nodes_in_partial_graph

        raw_batch["correct_first_node_type_choices"].append(
            self.node_types_to_multi_hot(graph_sample.correct_first_node_type_choices)
        )

        # Now also store all relevant graph properties:
        for property_name in self._graph_property_names:
            property_value = graph_sample.graph_property_values.get(property_name)
            if property_value is not None:
                raw_batch["graph_property_to_values"][property_name].append(property_value)
                raw_batch["graph_property_to_graph_ids"][property_name].append(
                    original_graph_id_in_batch
                )

    @staticmethod
    def _generate_self_loops(num_nodes: int) -> np.ndarray:
        """Generate a (num_nodes, 2) array of self loop edges."""
        return np.repeat(np.arange(num_nodes, dtype=np.int32), 2).reshape(-1, 2)

    def _finalise_batch(self, raw_batch: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        batch_features, batch_labels = super()._finalise_batch(raw_batch)

        # Features first.
        batch_features["node_categorical_features"] = safe_make_array(
            raw_batch["node_categorical_features"], dtype=np.int32
        )

        batch_features["node_to_partial_graph_map"] = safe_concat(
            raw_batch["node_to_partial_graph_map"], dtype=np.int32
        )

        batch_features["focus_nodes"] = safe_make_array(raw_batch["focus_nodes"], dtype=np.int32)

        batch_features["partial_node_features"] = safe_concat(
            raw_batch["partial_node_features"],
            dtype=np.float32,
            shape_suffix=self.partial_node_feature_shape,
        )

        batch_features["partial_node_categorical_features"] = np.array(
            raw_batch["partial_node_categorical_features"], dtype=np.int32
        )

        for i, adjacency_list in enumerate(raw_batch["partial_adjacency_lists"]):
            batch_features[f"partial_adjacency_list_{i}"] = safe_concat(
                adjacency_list, dtype=np.int32, shape_suffix=(2,)
            )

        # Add self loops if needed:
        if self._params["add_self_loop_edges"]:
            num_nodes = raw_batch["num_nodes_in_batch"]
            num_partial_nodes = raw_batch["num_partial_nodes_in_batch"]
            adjacency_list_idx = len(raw_batch["partial_adjacency_lists"])
            batch_features[f"adjacency_list_{adjacency_list_idx}"] = self._generate_self_loops(
                num_nodes
            )
            batch_features[
                f"partial_adjacency_list_{adjacency_list_idx}"
            ] = self._generate_self_loops(num_partial_nodes)

        batch_features["partial_node_to_original_node_map"] = safe_concat(
            raw_batch["partial_node_to_original_node_map"], dtype=np.int32
        )

        batch_features["partial_graph_to_original_graph_map"] = safe_make_array(
            raw_batch["partial_graph_to_original_graph_map"], dtype=np.int32
        )

        batch_features["num_partial_graphs_in_batch"] = raw_batch["num_partial_graphs_in_batch"]

        batch_features["partial_graphs_requiring_node_choices"] = safe_make_array(
            raw_batch["partial_graphs_requiring_node_choices"], dtype=np.int32
        )

        batch_features["valid_edge_choices"] = safe_concat(
            raw_batch["valid_edge_choices"], dtype=np.int32, shape_suffix=(2,)
        )

        batch_features["valid_attachment_point_choices"] = safe_concat(
            raw_batch["valid_attachment_point_choices"], dtype=np.int32
        )

        batch_features["edge_features"] = safe_concat(
            raw_batch["edge_features"], dtype=np.float32, shape_suffix=(3,)
        )

        # Now labels.
        batch_labels["node_types"] = np.array(raw_batch["node_types"], dtype=np.int32)

        batch_labels["correct_edge_choices"] = safe_concat(
            raw_batch["correct_edge_choices"], dtype=np.float32
        )

        batch_labels["num_correct_edge_choices"] = safe_make_array(
            raw_batch["num_correct_edge_choices"], dtype=np.int32
        )

        batch_labels["correct_attachment_point_choices"] = safe_make_array(
            raw_batch["correct_attachment_point_choices"], dtype=np.int32
        )

        batch_labels["stop_node_label"] = np.array(raw_batch["stop_node_label"], dtype=np.int32)

        batch_labels["correct_edge_types"] = safe_concat(
            raw_batch["correct_edge_types"],
            dtype=np.int32,
            shape_suffix=(self.num_edge_types_to_classify,),
        )

        batch_labels["valid_edge_types"] = safe_concat(
            raw_batch["valid_edge_types"],
            dtype=np.int32,
            shape_suffix=(self.num_edge_types_to_classify,),
        )

        batch_labels["correct_node_type_choices"] = safe_make_array(
            raw_batch["correct_node_type_choices"],
            dtype=np.float32,
            shape_suffix=(self.num_node_types,),
        )

        batch_labels["correct_first_node_type_choices"] = safe_make_array(
            raw_batch["correct_first_node_type_choices"],
            dtype=np.float32,
            shape_suffix=(self.num_node_types,),
        )

        # Now also store all relevant graph properties:
        batch_features[f"graph_properties_present"] = []
        for property_name in self._graph_property_names:
            property_values = raw_batch["graph_property_to_values"][property_name]
            if len(property_values) > 0:
                batch_features[f"graph_properties_present"].append(property_name)
                batch_labels[f"graph_property_{property_name}_values"] = property_values
                batch_features[f"graph_property_{property_name}_graph_ids"] = raw_batch[
                    "graph_property_to_graph_ids"
                ][property_name]

        return batch_features, batch_labels

    def get_batch_tf_data_description(self) -> GraphBatchTFDataDescription:
        base_description = super().get_batch_tf_data_description()
        batch_features_types = base_description.batch_features_types
        batch_features_shapes = base_description.batch_features_shapes
        batch_labels_types = base_description.batch_labels_types
        batch_labels_shapes = base_description.batch_labels_shapes

        # Features first.
        batch_features_types["node_categorical_features"] = tf.int32
        batch_features_shapes["node_categorical_features"] = (None,)
        batch_features_types["node_to_partial_graph_map"] = tf.int32
        batch_features_shapes["node_to_partial_graph_map"] = (None,)
        batch_features_types["focus_nodes"] = tf.int32
        batch_features_shapes["focus_nodes"] = (None,)
        batch_features_types["partial_node_features"] = tf.float32
        batch_features_shapes["partial_node_features"] = (None,) + self.partial_node_feature_shape
        batch_features_types["partial_node_categorical_features"] = tf.int32
        batch_features_shapes["partial_node_categorical_features"] = (None,)
        for edge_type_idx in range(self.num_edge_types):
            batch_features_types[f"partial_adjacency_list_{edge_type_idx}"] = tf.int32
            batch_features_shapes[f"partial_adjacency_list_{edge_type_idx}"] = (None, 2)
        batch_features_types["partial_node_to_original_node_map"] = tf.int32
        batch_features_shapes["partial_node_to_original_node_map"] = (None,)
        batch_features_types["partial_graph_to_original_graph_map"] = tf.int32
        batch_features_shapes["partial_graph_to_original_graph_map"] = (None,)
        batch_features_types["num_partial_graphs_in_batch"] = tf.int32
        batch_features_shapes["num_partial_graphs_in_batch"] = ()
        batch_features_types["partial_graphs_requiring_node_choices"] = tf.int32
        batch_features_shapes["partial_graphs_requiring_node_choices"] = (None,)
        batch_features_types["valid_edge_choices"] = tf.int32
        batch_features_shapes["valid_edge_choices"] = (None, 2)
        batch_features_types["edge_features"] = tf.float32
        batch_features_shapes["edge_features"] = (None, 3)
        batch_features_types["valid_attachment_point_choices"] = tf.int32
        batch_features_shapes["valid_attachment_point_choices"] = (None,)

        # Now labels.
        batch_labels_types["node_types"] = tf.int32
        batch_labels_shapes["node_types"] = (None,)
        batch_labels_types["correct_edge_choices"] = tf.float32
        batch_labels_shapes["correct_edge_choices"] = (None,)
        batch_labels_types["num_correct_edge_choices"] = tf.int32
        batch_labels_shapes["num_correct_edge_choices"] = (None,)
        batch_labels_types["stop_node_label"] = tf.int32
        batch_labels_shapes["stop_node_label"] = (None,)
        batch_labels_types["correct_edge_types"] = tf.int32
        batch_labels_shapes["correct_edge_types"] = (None, self.num_edge_types_to_classify)
        batch_labels_types["valid_edge_types"] = tf.int32
        batch_labels_shapes["valid_edge_types"] = (None, self.num_edge_types_to_classify)
        batch_labels_types["correct_attachment_point_choices"] = tf.int32
        batch_labels_shapes["correct_attachment_point_choices"] = (None,)
        batch_labels_types["correct_node_type_choices"] = tf.float32
        batch_labels_shapes["correct_node_type_choices"] = (None, self.num_node_types)
        batch_labels_types["correct_first_node_type_choices"] = tf.float32
        batch_labels_shapes["correct_first_node_type_choices"] = (None, self.num_node_types)

        # Declare information about potentially provided graph properties and their values.
        batch_features_types["graph_properties_present"] = tf.string
        batch_features_shapes["graph_properties_present"] = (None,)

        for property_name in self._graph_property_names:
            batch_features_types[f"graph_property_{property_name}_graph_ids"] = tf.int32
            batch_features_shapes[f"graph_property_{property_name}_graph_ids"] = (None,)
            batch_labels_types[f"graph_property_{property_name}_values"] = tf.float32
            batch_labels_shapes[f"graph_property_{property_name}_values"] = (None,)

        return GraphBatchTFDataDescription(
            batch_features_types=batch_features_types,
            batch_features_shapes=batch_features_shapes,
            batch_labels_types=batch_labels_types,
            batch_labels_shapes=batch_labels_shapes,
        )

    @abstractmethod
    def load_data(self, path, folds_to_load):
        pass

    @abstractmethod
    def node_type_to_index(self, node_type: str) -> int:
        """Convert between string representation and integer index.

        Implementations of this method must take a string representation of the node type (e.g.,
        "C", "F", "Cl" for atom types, for example) and return a vocabulary index.
        """
        raise NotImplementedError

    def node_types_to_indices(self, node_types: List[str]) -> List[int]:
        """Convert list of string representations into list of integer indices."""
        return [self.node_type_to_index(node_type) for node_type in node_types]

    def node_types_to_multi_hot(self, node_types: List[str]) -> np.ndarray:
        """Convert between string representation to multi hot encoding of correct node types.

        Note: implemented here for backwards compatibility only.
        """
        return np.zeros(shape=(self.num_node_types,), dtype=np.float32)

    @property
    def num_edge_types(self) -> int:
        return self._num_edge_types

    @property
    @abstractmethod
    def node_type_index_to_string(self) -> Dict[int, str]:
        """Return a dictionary to go from node type index to the string representation.

        This is a dictionary which is the inverse function to node_type_to_index.
        For example, if self.node_type_to_index(["C", "F"]) returned [3, 5], then the
        returned dictionary should map 3 to "C" and 5 to "F".

        Returns:
            A dictionary of node type indices to string representations of those nodes.

        """
        raise NotImplementedError

    def _load_one_sample(self, data_fold: DataFold):
        graph_it = self._graph_iterator(data_fold)
        return next(iter(graph_it))

    def _load_feature_shapes_from_data(self):
        assert (
            len(self._loaded_data) > 0
        ), "We must have loaded some data before we can calculate the node feature shape."
        data_fold = next(iter(self._loaded_data.keys()))
        datum = self._load_one_sample(data_fold)

        setattr(self, "_node_feature_shape", (len(datum.node_features[0]),))
        setattr(self, "_partial_node_feature_shape", (len(datum.partial_node_features[0][0]),))
        setattr(self, "_node_categorical_num_classes", datum.node_categorical_num_classes)

        self.metadata["_node_categorical_num_classes"] = self._node_categorical_num_classes

    def _get_cached_property(self, name: str) -> Any:
        if not self._loaded_feature_shapes:
            self._load_feature_shapes_from_data()
            self._loaded_feature_shapes = True

        return getattr(self, name, None)

    @property
    def node_feature_shape(self) -> Tuple:
        return self._get_cached_property("_node_feature_shape")

    @property
    def partial_node_feature_shape(self) -> Tuple:
        return self._get_cached_property("_partial_node_feature_shape")

    @property
    def node_categorical_num_classes(self) -> Optional[int]:
        return self._get_cached_property("_node_categorical_num_classes")

    @property
    def uses_categorical_features(self) -> bool:
        return self.node_categorical_num_classes is not None

    @property
    def num_node_types(self) -> int:
        return len(self.node_type_index_to_string)

    def cleanup_dataset_resources(self, dataset_to_cleanup: tf.data.Dataset) -> None:
        """
        Call this function to mark that a given dataset is not required anymore. In some
        instances, the TraceDataset class creates substantial resources (processes, file handles)
        that should be cleaned up, but that doesn't always happen when the tf.data.Dataset
        goes out of scope.

        NB: This _should_ be handled in the style of a context manager, but would require
        changes in a huge number of places, hence this (at most) second-best solution...
        """
        # No-op in cases where the dataset iterators are trivial
        pass


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
