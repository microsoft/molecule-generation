from typing import List, Tuple

import numpy as np


def safe_concat(data: List[np.ndarray], dtype: np.dtype, shape_suffix: Tuple[int, ...] = ()):
    if len(data) > 0:
        return np.concatenate(data).astype(dtype)
    else:
        return np.zeros(shape=(0,) + shape_suffix, dtype=dtype)


def safe_make_array(data: List[np.ndarray], dtype: np.dtype, shape_suffix: Tuple[int, ...] = ()):
    if len(data) > 0:
        return np.array(data, dtype=dtype)
    else:
        return np.zeros(shape=(0,) + shape_suffix, dtype=dtype)
