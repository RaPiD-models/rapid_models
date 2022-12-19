# import pytest
import numpy as np

from rapid_models.gp_diagnostics.utils.checks import (
    is_numeric_np_array,
    is_square,
    is_lower_triang,
)


def test_is_numeric_np_array():
    """
    Test the function is_numeric_np_array()
    """

    # These should be ok
    assert is_numeric_np_array(np.array([1, 2, 2.3]))
    assert is_numeric_np_array(np.array([[1, 2, 2.3], [0, 1e-6, 0.001]]))
    assert is_numeric_np_array(np.array(200))

    # These are not ok
    # assert is_numeric_np_array(np.array([[1, 2, 2.3], [0, 0.001]])) is False
    assert is_numeric_np_array(np.array(None)) is False
    assert is_numeric_np_array(np.array([[1, 2, 2.3], [0, 0.001, "a"]])) is False
    assert is_numeric_np_array("a") is False
    assert is_numeric_np_array(50) is False
    assert is_numeric_np_array([1, 2, 3]) is False


def test_is_square():
    """
    Test the function is_square()
    """
    # These should be ok
    assert is_square(np.ones(shape=(1, 1)))
    assert is_square(np.ones(shape=(14, 14)))

    # These are not ok
    assert is_square(np.ones(shape=(13, 14))) is False
    assert is_square(np.ones(shape=(3, 2))) is False
    assert is_square(np.ones(shape=(3, 3, 3))) is False
    assert is_square(np.array(2)) is False
    assert is_square(np.array([1, 2, 3])) is False


def test_is_lower_triang():
    """
    Test the function is_lower_triang()
    """
    # These should be ok
    arr = np.array([[0.2, 0, 0], [3, 2.2, 0], [1, 2, 4]])

    assert is_lower_triang(arr)

    arr = np.array([[1, 0], [2, 2.2]])
    assert is_lower_triang(arr)

    arr = np.array([[1]])
    assert is_lower_triang(arr)

    # These are not ok
    arr = np.array([[1, 2, 2.3], [0, 2.2, 3], [0.1, 0, 4]])

    assert is_lower_triang(arr) is False
