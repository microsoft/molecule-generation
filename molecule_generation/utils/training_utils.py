import numpy as np
from typing import List


SMALL_NUMBER = 1e-7


def get_class_balancing_weights(class_counts: List[int], class_weight_factor: float) -> np.ndarray:
    class_counts = np.array(class_counts, dtype=np.float32)
    total_count = np.sum(class_counts)

    class_occ_weights = (total_count / (class_counts + SMALL_NUMBER)) / class_counts.shape[0]

    class_occ_weights = np.clip(class_occ_weights, a_min=0.1, a_max=10)
    class_std_weights = np.ones(class_occ_weights.shape)

    class_weights = (
        class_weight_factor * class_occ_weights + (1 - class_weight_factor) * class_std_weights
    ).astype(np.float32)

    return class_weights
