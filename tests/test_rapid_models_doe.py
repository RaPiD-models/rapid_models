#!/usr/bin/env python

"""Tests for `rapid_models.doe` package."""

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
        doe.fullfact_with_bounds([0.1, 0.9], [0.1, 0.9], [-1, 5])


def test_fullfact_with_bounds_zero_dim():
    """
    Test ValueError raised for zero dimension
    """
    with pytest.raises(ValueError):
        doe.fullfact_with_bounds([0.1, 0.9], [0.1, 0.9], [3, 0])


def test_fullfact_with_bounds_input_shape1():
    """
    Test ValueError raised for different first dimension of LB
    """
    with pytest.raises(ValueError):
        doe.fullfact_with_bounds([0.1], [0.1, 0.9], [3, 3])


def test_fullfact_with_bounds_input_shape2():
    """
    Test ValueError raised for different first dimension of UB
    """
    with pytest.raises(ValueError):
        doe.fullfact_with_bounds([0.1, 0.9], [0.1, 0.9, 1.2], [3, 3])


def test_fullfact_with_bounds_input_shape3():
    """
    Test ValueError raised for different first dimension of N_xi
    """
    with pytest.raises(ValueError):
        doe.fullfact_with_bounds([0.1, 0.9], [0.1, 0.9], [3, 1, 4])


def test_fullfact_with_bounds_input_shape4():
    """
    Test ValueError raised for input dimensions > 1
    """
    with pytest.raises(ValueError):
        doe.fullfact_with_bounds([[0.1, 0.9]], [[0.1, 0.9]], [[3, 1]])


def test_fullfact_with_bounds_1():
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


# Tests related to rapid_models.doe.lhs_with_bounds()


def test_lhs_with_bounds():
    """Sample pytest test function with the pytest fixture as an argument. what should we write to make this line too"""
    assert 1 == 1  # is None
