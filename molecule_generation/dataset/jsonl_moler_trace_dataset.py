"""Class handling MoLeR trace datasets."""
import random
from typing import Any, Dict, List, Optional, Set

import numpy as np
from molecule_generation.dataset.jsonl_abstract_trace_dataset import JSONLAbstractTraceDataset
from molecule_generation.dataset.trace_sample import TraceStep
from molecule_generation.chem.motif_utils import get_motif_type_to_node_type_index_map
from dpu_utils.utils import RichPath
from tf2_gnn.data.graph_dataset import DataFold


class JSONLMoLeRTraceDataset(JSONLAbstractTraceDataset):
    """JSONLMoLeRTraceDataset"""

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        super_params = super().get_default_hyperparameters()
        these_hypers: Dict[str, Any] = {
            "trace_element_keep_prob": 0.4,
            "trace_element_non_carbon_keep_prob": 0.5,
        }
        super_params.update(these_hypers)
        return super_params

    def __init__(
        self,
        params: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        no_parallelism: bool = False,
    ):
        super().__init__(params, metadata=metadata, no_parallelism=no_parallelism)
        assert params["add_self_loop_edges"], "MoLeR requires the addition of self-loop edges."

    def load_data(self, path: RichPath, folds_to_load: Optional[Set[DataFold]] = None) -> None:
        super().load_data(path, folds_to_load)

        self._motif_vocabulary = self.metadata.get("motif_vocabulary")

        if self._motif_vocabulary is not None:
            self._motif_to_node_type_index = get_motif_type_to_node_type_index_map(
                motif_vocabulary=self._motif_vocabulary,
                num_atom_types=len(self._node_type_index_to_string),
            )

            for motif, node_type in self._motif_to_node_type_index.items():
                self._node_type_index_to_string[node_type] = motif
        else:
            self._motif_to_node_type_index = {}

    def node_type_to_index(self, node_type: str) -> int:
        motif_node_type_index = self._motif_to_node_type_index.get(node_type)

        if motif_node_type_index is not None:
            return motif_node_type_index
        else:
            return super().node_type_to_index(node_type)

    def node_types_to_multi_hot(self, node_types: List[str]) -> np.ndarray:
        """Convert between string representation to multi hot encoding of correct node types."""
        # This lives here (not in superclass) as it's not applicable to the CGVAE trace dataset.
        correct_indices = self.node_types_to_indices(node_types)
        multihot = np.zeros(shape=(self.num_node_types,), dtype=np.float32)
        for idx in correct_indices:
            multihot[idx] = 1.0
        return multihot

    def _include_trace_step_in_batch(self, trace_step: TraceStep) -> bool:
        # Optionally protect all partial graphs corresponding to the more unusual node
        # choices related to non-carbons (effectively, oversample subgraphs in which
        # we need to choose nodes and carbon is not a correct option):
        if (
            trace_step.correct_node_type_choices is not None
            and "C" not in trace_step.correct_node_type_choices
        ):
            return random.uniform(0, 1) < self._params["trace_element_non_carbon_keep_prob"]

        # Otherwise, fall back onto default sampling behaviour:
        return super()._include_trace_step_in_batch(trace_step)
