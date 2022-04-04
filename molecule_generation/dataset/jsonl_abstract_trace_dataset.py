"""Abstract superclass of trace dataset."""
import logging
from typing import Any, Dict, List, Optional, Set

import tensorflow as tf
from dpu_utils.utils import RichPath
from tf2_gnn import DataFold

from molecule_generation.dataset.trace_dataset import TraceDataset, TraceSample
from molecule_generation.utils.sharded_data_reader import ShardedDataReader
from molecule_generation.chem.atom_feature_utils import AtomTypeFeatureExtractor

logger = logging.getLogger(__name__)


class ContextManagedTfDataset:
    def __init__(self, tf_dataset: tf.data.Dataset, enter_fn=None, exit_fn=None):
        def default_entry_fn(self):
            return self

        def default_exit_fn(self, exc_type, exc_value, traceback):
            return False

        self._enter_fn = enter_fn or default_entry_fn
        self._exit_fn = exit_fn or default_exit_fn
        self.tf_dataset = tf_dataset

    def __enter__(self):
        return self._enter_fn(self)

    def __exit__(self, exc_type, exc_value, traceback):
        return self._exit_fn(self, exc_type, exc_value, traceback)


class JSONLAbstractTraceDataset(TraceDataset):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        super_params = super().get_default_hyperparameters()
        these_hypers: Dict[str, Any] = {
            "data_reader_max_queue_size": 500,
            "data_reader_num_workers": 20,
        }
        super_params.update(these_hypers)
        return super_params

    def __init__(
        self,
        params: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        no_parallelism: bool = False,
    ):
        super().__init__(params, metadata=metadata)
        self._atom_type_featuriser: AtomTypeFeatureExtractor = None
        # Overridden from base class because of type difference.
        self._loaded_data: Dict[DataFold, List[RichPath]] = {}
        self._no_parallelism = no_parallelism

    def load_data(self, path: RichPath, folds_to_load: Optional[Set[DataFold]] = None) -> None:
        logger.info(f"Loading data from {path}.")

        if not self._metadata:
            metadata_path = path.join("metadata.pkl.gz")
            logger.info(f"Loading metadata from {metadata_path}")
            self._metadata = metadata_path.read_by_file_suffix()
        else:
            logger.warning("Using metadata passed to constructor, not that saved with the dataset.")

        self._atom_type_featuriser = next(
            featuriser
            for featuriser in self._metadata["feature_extractors"]
            if featuriser.name == "AtomType"
        )

        self._node_type_index_to_string = self._atom_type_featuriser.index_to_atom_type_map.copy()

        if folds_to_load is None:
            folds_to_load = {DataFold.TRAIN, DataFold.TEST, DataFold.VALIDATION}

        def find_fold_files(directory: RichPath, fold_prefix: str) -> List[RichPath]:
            """Find all file paths in directory that start with fold_prefix."""
            return [
                path
                for path in directory.get_filtered_files_in_dir(f"**/{fold_prefix}*.pkl.gz")
                if path.is_file()
            ]

        if DataFold.TRAIN in folds_to_load:
            self._loaded_data[DataFold.TRAIN] = find_fold_files(path, "train")
            logger.debug(f"Loaded {len(self._loaded_data[DataFold.TRAIN])} training shards.")

        if DataFold.TEST in folds_to_load:
            self._loaded_data[DataFold.TEST] = find_fold_files(path, "test")
            logger.debug(f"Loaded {len(self._loaded_data[DataFold.TEST])} test shards.")

        if DataFold.VALIDATION in folds_to_load:
            self._loaded_data[DataFold.VALIDATION] = find_fold_files(path, "valid")
            logger.debug(f"Loaded {len(self._loaded_data[DataFold.VALIDATION])} validation shards.")

    def node_type_to_index(self, node_type: str) -> int:
        return self._atom_type_featuriser.type_name_to_index(node_type)

    @property
    def node_type_index_to_string(self) -> Dict[int, str]:
        return self._node_type_index_to_string

    def _load_one_sample(self, data_fold: DataFold):
        graph_it = self._graph_iterator(data_fold, no_parallelism=True)
        return next(iter(graph_it))

    def _graph_iterator(
        self, data_fold: DataFold, no_parallelism: bool = False
    ) -> ShardedDataReader[TraceSample]:
        sharded_data_reader = ShardedDataReader(
            self._loaded_data[data_fold],
            max_queue_size=self._params["data_reader_max_queue_size"],
            num_workers=self._params["data_reader_num_workers"],
            shuffle_data=data_fold == DataFold.TRAIN,
            repeat_data=data_fold == DataFold.TRAIN,
            no_parallelism=no_parallelism or self._no_parallelism,
        )
        return sharded_data_reader

    def get_tensorflow_dataset(self, data_fold: DataFold) -> tf.data.Dataset:
        return self.get_context_managed_tf_dataset(data_fold).tf_dataset

    def get_context_managed_tf_dataset(
        self,
        data_fold: DataFold,
    ) -> ContextManagedTfDataset:
        """Construct a TensorFlow dataset for the specified data fold.

        Returns:
            A tensorflow Dataset object. Each element in the dataset is a pair of
            dictionaries representing features and labels.
            The content of these is determined by the _finalise_batch method.
        """
        data_description = self.get_batch_tf_data_description()
        graph_sample_iterator = self._graph_iterator(data_fold)

        dataset = tf.data.Dataset.from_generator(
            generator=lambda: self.graph_batch_iterator_from_graph_iterator(graph_sample_iterator),
            output_types=(
                data_description.batch_features_types,
                data_description.batch_labels_types,
            ),
            output_shapes=(
                data_description.batch_features_shapes,
                data_description.batch_labels_shapes,
            ),
        )
        dataset = dataset.prefetch(5)

        def dataset_context_exit_fn(self, exc_type, exc_value, traceback):
            return graph_sample_iterator.__exit__(exc_type, exc_value, traceback)

        return ContextManagedTfDataset(dataset, exit_fn=dataset_context_exit_fn)
