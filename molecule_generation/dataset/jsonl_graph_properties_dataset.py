"""General dataset class for datasets with several properties stored as JSONLines files."""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from dpu_utils.utils import RichPath
from tf2_gnn.data.graph_dataset import GraphBatchTFDataDescription, GraphSample
from tf2_gnn.data.jsonl_graph_dataset import JsonLGraphDataset

from molecule_generation.utils.property_models import PropertyTaskType


class GraphWithPropertiesSample(GraphSample):
    """Data structure holding a single graph with a list of numeric properties."""

    def __init__(
        self,
        adjacency_lists: List[np.ndarray],
        type_to_node_to_num_incoming_edges: np.ndarray,
        node_features: List[np.ndarray],
        target_values: List[float],
    ):
        super().__init__(adjacency_lists, type_to_node_to_num_incoming_edges, node_features)
        self._target_values = target_values

    @property
    def target_values(self) -> List[float]:
        """Target value of the regression tasks."""
        return self._target_values

    def __str__(self):
        return (
            f"Adj:            {self._adjacency_lists}\n"
            f"Node_features:  {self._node_features}\n"
            f"Target_values:  {self._target_values}"
        )


class JsonLGraphPropertiesDataset(JsonLGraphDataset[GraphWithPropertiesSample]):
    """
    General class representing pre-split datasets in JSONLines format.
    Concretely, this class expects the following:
    * In the data directory, files "train.jsonl.gz", "valid.jsonl.gz" and
      "test.jsonl.gz" are used to store the train/valid/test datasets.
    * Either a file "metadata.pkl.gz" in the data directory, or a value for
      the constructor argument metadata, containing a dictionary which at
      least has to contain the key "property_names" with a list of names
      of the properties stored for the data.
      The names used in the "properties_to_use" hyperparameter need to be
      a subset of this list.
    * Each of the files is gzipped text file in which each line is a valid
      JSON dictionary with a "graph" key, which in turn points to a
      dictionary with keys
       - "node_features" (list of numerical initial node labels)
       - "adjacency_lists" (list of list of directed edge pairs)
      Addtionally, the dictionary has to contain a "properties" key with a
      list of floating point values (the number of values needs to match
      the number of "property_names" provided in the metadata)
    """

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        super_hypers = super().get_default_hyperparameters()
        this_hypers: Dict[str, Any] = {
            # If None, the data-provided property is used; otherwise,
            # either a floating point value is expected and property values
            # greater than this value will be encoded as 1.0 and smaller
            # values will be encoded as 0.0; or a dictionary mapping property
            # names to specific thresholds (if a used property does not appear,
            # it will not be transformed into 0.0/1.0).
            "thresholds_for_classification": None,
            # List of property names to expose as labels to the downstream models.
            "properties_to_use": [],
            # Associates property names to task types; if not given, assumes that
            # thresholded properties are treated as binary classification tasks
            # and all others are regression tasks.
            # Currently supports "binary_classification" and "regression".
            "property_name_to_type": {},
        }
        super_hypers.update(this_hypers)
        return super_hypers

    def __init__(
        self,
        params: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(params, metadata=metadata)
        self._property_names = params["properties_to_use"]
        self._num_properties = len(self._property_names)
        self._property_name_to_dataset_index: Dict[str, int] = {}  # Filled when loading data.
        self._thresholds_for_classification: Dict[str, float] = {}
        self._property_name_to_type: Dict[str, PropertyTaskType] = {}

    def load_metadata(self, path: RichPath) -> None:
        super().load_metadata(path)

        if len(self._property_names) == 0:
            print(f"W: No properties to operate on selected.")
            print(f"   Using all present in dataset: {self.metadata['property_names']}")
            self._property_names = self.metadata["property_names"]
            self._num_properties = len(self._property_names)

        if isinstance(self.params["thresholds_for_classification"], float):
            for prop_name in self._property_names:
                self._thresholds_for_classification[prop_name] = self.params[
                    "thresholds_for_classification"
                ]
        for prop_name in self._property_names:
            prop_type: Optional[str] = self.params["property_name_to_type"].get(prop_name)
            if prop_type is not None:
                self._property_name_to_type[prop_name] = PropertyTaskType[prop_type.upper()]
            elif prop_name in self._thresholds_for_classification:
                self._property_name_to_type[prop_name] = PropertyTaskType.BINARY_CLASSIFICATION
            else:
                self._property_name_to_type[prop_name] = PropertyTaskType.REGRESSION

        # Some datasets (dense multitask) have a list of property names in the metadata and
        # only store actual values as a list with each datum. Store the mapping from name to
        # list index here:
        if "property_names" in self.metadata:
            self._property_name_to_dataset_index = {
                prop: idx for idx, prop in enumerate(self.metadata["property_names"])
            }

    def _process_raw_datapoint(self, datapoint: Dict[str, Any]) -> GraphWithPropertiesSample:
        node_features = datapoint["graph"]["node_features"]
        type_to_adj_list, type_to_num_incoming_edges = self._process_raw_adjacency_lists(
            raw_adjacency_lists=datapoint["graph"]["adjacency_lists"],
            num_nodes=len(node_features),
        )

        target_values = []
        for prop_name in self._property_names:
            # In the dense multitask setting, resolve property name into an index and get value:
            if isinstance(datapoint["properties"], list):
                property_index = self._property_name_to_dataset_index[prop_name]
                target_value = datapoint["properties"][property_index]
            else:
                target_value = datapoint["properties"][prop_name]

            threshold = self._thresholds_for_classification.get(prop_name)
            if threshold is not None:
                target_value = float(target_value > threshold)

            target_values.append(target_value)

        return GraphWithPropertiesSample(
            adjacency_lists=type_to_adj_list,
            type_to_node_to_num_incoming_edges=type_to_num_incoming_edges,
            node_features=node_features,
            target_values=target_values,
        )

    def _new_batch(self) -> Dict[str, Any]:
        new_batch = super()._new_batch()
        new_batch["target_values"] = []
        return new_batch

    def _add_graph_to_batch(
        self, raw_batch: Dict[str, Any], graph_sample: GraphWithPropertiesSample
    ) -> None:
        super()._add_graph_to_batch(raw_batch, graph_sample)
        raw_batch["target_values"].append(graph_sample.target_values)

    def _finalise_batch(self, raw_batch) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        batch_features, batch_labels = super()._finalise_batch(raw_batch)
        return batch_features, {"target_values": raw_batch["target_values"]}

    def get_batch_tf_data_description(self) -> GraphBatchTFDataDescription:
        data_description = super().get_batch_tf_data_description()
        return GraphBatchTFDataDescription(
            batch_features_types=data_description.batch_features_types,
            batch_features_shapes=data_description.batch_features_shapes,
            batch_labels_types={**data_description.batch_labels_types, "target_values": tf.float32},
            batch_labels_shapes={
                **data_description.batch_labels_shapes,
                "target_values": (None, self._num_properties),
            },
        )
