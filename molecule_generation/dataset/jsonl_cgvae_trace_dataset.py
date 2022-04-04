"""Class handling CGVAE trace datasets."""
import logging
from typing import Any, Dict

from molecule_generation.dataset.jsonl_abstract_trace_dataset import JSONLAbstractTraceDataset

logger = logging.getLogger(__name__)


class JSONLCGVAETraceDataset(JSONLAbstractTraceDataset):
    """JSONLCGVAETraceDataset"""

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        super_params = super().get_default_hyperparameters()
        these_hypers: Dict[str, Any] = {}
        super_params.update(these_hypers)
        return super_params
