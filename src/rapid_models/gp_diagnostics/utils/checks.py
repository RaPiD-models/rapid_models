# Div functions for argument checking

from typing import Any

import numpy as np
from nptyping import NDArray, Shape


def is_numeric_np_array(arr: NDArray[Any, Any]) -> bool:
    """
    Check that arr is a numpy array with only numeric elements
    """
    if not isinstance(arr, np.ndarray):
        return False
    return arr.dtype.kind in set(
        "buifc"
    )  # Boolean, unsigned integer, signed integer, float, complex.


def is_square(arr: NDArray[Shape["N, N"], Any]) -> bool:  # noqa: F821
    """
    Check that the numpy array arr is 2d quare
    """
    if len(arr.shape) != 2:
        return False
    return arr.shape[0] == arr.shape[1]


def is_lower_triang(arr: NDArray[Shape["N, N"], Any]) -> bool:  # noqa: F821
    """
    Check that that a square 2d numpy array is lower triangular
    """
    idx = np.triu_indices_from(arr, k=1)
    return all(arr[idx] == np.zeros(arr[idx].shape[0]))
