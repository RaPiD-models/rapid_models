#!/usr/bin/env python

"""Tests for `rapid_models.doe` module."""

import pytest

import rapid_models.doe as doe

import numpy as np

# 1. Arrange
# 2. Act
# 3. Assert
# 4. Cleanup

# Tests related to rapid_models.doe.fullfact_with_bounds()


def test_fullfact_with_bounds_pos_dim():
    """
    Test ValueError raised for negative dimensions
    """
    with pytest.raises(ValueError):
        doe.fullfact_with_bounds([0.1, 0.1], [0.9, 0.9], [-1, 5])


def test_fullfact_with_bounds_zero_dim():
    """
    Test ValueError raised for zero dimension
    """
    with pytest.raises(ValueError):
        doe.fullfact_with_bounds([0.1, 0.1], [0.9, 0.9], [3, 0])


def test_fullfact_with_bounds_input_shape1():
    """
    Test ValueError raised for different first dimension of LB
    """
    with pytest.raises(ValueError):
        doe.fullfact_with_bounds([0.1], [0.9, 0.9], [3, 3])


def test_fullfact_with_bounds_input_shape2():
    """
    Test ValueError raised for different first dimension of UB
    """
    with pytest.raises(ValueError):
        doe.fullfact_with_bounds([0.1, 0.1], [0.9, 0.9, 1.2], [3, 3])


def test_fullfact_with_bounds_input_shape3():
    """
    Test ValueError raised for different first dimension of N_xi
    """
    with pytest.raises(ValueError):
        doe.fullfact_with_bounds([0.1, 0.1], [0.9, 0.9], [3, 1, 4])


def test_fullfact_with_bounds_input_shape4():
    """
    Test ValueError raised for input dimensions > 1
    """
    with pytest.raises(ValueError):
        doe.fullfact_with_bounds([[0.1, 0.9]], [[0.1, 0.9]], [[3, 1]])


def test_fullfact_with_bounds_output_shape1():
    """
    Test output shape 1
    """
    assert (3 * 5, 2) == doe.fullfact_with_bounds([0.1, 0.1], [0.9, 0.9], [3, 5]).shape


def test_fullfact_with_bounds_1():
    """
    Test output 1
    """
    # 1. + 2.
    X = np.asarray(doe.fullfact_with_bounds([0.1, 0.1], [0.9, 0.9], [3, 5]))
    X = X[X[:, 0].argsort()]
    # 3.
    assert np.allclose(
        X,
        np.array(
            [
                [0.1, 0.1],
                [0.1, 0.3],
                [0.1, 0.5],
                [0.1, 0.7],
                [0.1, 0.9],
                [0.5, 0.1],
                [0.5, 0.3],
                [0.5, 0.5],
                [0.5, 0.7],
                [0.5, 0.9],
                [0.9, 0.1],
                [0.9, 0.3],
                [0.9, 0.5],
                [0.9, 0.7],
                [0.9, 0.9],
            ]
        ),
    )


def test_fullfact_with_bounds_2():
    """
    Test output 2
    """
    # 1. + 2.
    X = np.asarray(doe.fullfact_with_bounds([-3, 4], [5, 12], [5, 3]))
    X = X[X[:, 0].argsort()]
    # 3.
    assert np.allclose(
        X,
        np.array(
            [
                [-3.0, 4.0],
                [-3.0, 8.0],
                [-3.0, 12.0],
                [-1.0, 4.0],
                [-1.0, 8.0],
                [-1.0, 12.0],
                [1.0, 4.0],
                [1.0, 8.0],
                [1.0, 12.0],
                [3.0, 4.0],
                [3.0, 8.0],
                [3.0, 12.0],
                [5.0, 4.0],
                [5.0, 8.0],
                [5.0, 12.0],
            ]
        ),
    )


def test_fullfact_with_bounds_3():
    """
    Test output 3
    """
    # 1. + 2.
    X = np.asarray(
        doe.fullfact_with_bounds([0.1, 0.1, 0.1], [0.9, 0.9, 1.2], [3, 3, 2])
    )
    # X = X[X[:, 0].argsort()]
    # 3.
    assert np.allclose(
        X,
        np.array(
            [
                [0.1, 0.1, 0.1],
                [0.5, 0.1, 0.1],
                [0.9, 0.1, 0.1],
                [0.1, 0.5, 0.1],
                [0.5, 0.5, 0.1],
                [0.9, 0.5, 0.1],
                [0.1, 0.9, 0.1],
                [0.5, 0.9, 0.1],
                [0.9, 0.9, 0.1],
                [0.1, 0.1, 1.2],
                [0.5, 0.1, 1.2],
                [0.9, 0.1, 1.2],
                [0.1, 0.5, 1.2],
                [0.5, 0.5, 1.2],
                [0.9, 0.5, 1.2],
                [0.1, 0.9, 1.2],
                [0.5, 0.9, 1.2],
                [0.9, 0.9, 1.2],
            ]
        ),
    )


# Tests related to rapid_models.doe.lhs_with_bounds()


def test_lhs_with_bounds_input_shape1():
    """
    Test ValueError raised for different first dimension of LB
    """
    with pytest.raises(ValueError):
        doe.lhs_with_bounds(2, 15, [0.1], [0.9, 0.9])


def test_lhs_with_bounds_input_shape2():
    """
    Test ValueError raised for different first dimension of UB
    """
    with pytest.raises(ValueError):
        doe.lhs_with_bounds(2, 15, [0.1, 0.1], [0.9, 0.9, 1.2])


def test_lhs_with_bounds_input_shape3():
    """
    Test ValueError raised for different nDim than dimension of LB and UB
    """
    with pytest.raises(ValueError):
        doe.lhs_with_bounds(3, 15, [0.1, 0.1], [0.9, 0.9])


def test_lhs_with_bounds_output_shape1():
    """
    Test output shape
    """
    assert (15, 2) == doe.lhs_with_bounds(2, 15, [0.1, 0.1], [0.9, 0.9]).shape


def test_lhs_with_bounds_1():
    """
    Test output 1
    """
    X = doe.lhs_with_bounds(2, 15, [0.1, 0.1], [0.9, 0.9], 42)
    assert np.allclose(
        X,
        np.array(
            [
                [0.87826211, 0.64077301],
                [0.66596549, 0.55465368],
                [0.48303066, 0.84914402],
                [0.76432373, 0.59553222],
                [0.60303707, 0.82075917],
                [0.36776451, 0.78187605],
                [0.70224771, 0.15070476],
                [0.11997547, 0.41839519],
                [0.26309779, 0.21498637],
                [0.19237301, 0.48311491],
                [0.8039826, 0.43132475],
                [0.54289292, 0.70620596],
                [0.3453928, 0.30619606],
                [0.46439694, 0.18526179],
                [0.21498766, 0.3510972],
            ]
        ),
    )
