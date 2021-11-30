#!/usr/bin/env python

"""Tests for `rapid_models.doe` module."""

# import pytest

import rapid_models.doe.adaptive_learning as doe_al

import numpy as np

# 1. Arrange
# 2. Act
# 3. Assert
# 4. Cleanup

# Tests related to util function dotdot_a_b_aT_for_row_in_a(a,b)


def test_dotdot_a_b_aT_for_row_in_a():
    """
    Test output 1
    """
    # 1. + 2.
    a = [
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4],
        [5, 5, 5, 5, 5],
    ]
    b = [
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4],
        [5, 5, 5, 5, 5],
    ]
    X = doe_al.dotdot_a_b_aT_for_row_in_a(a, b)
    # 3.
    assert np.allclose(
        X,
        np.array([75, 300, 675, 1200, 1875]),
    )


# Tests related to util function dotdot_a_b_aT(a,b)


def test_dotdot_a_b_aT():
    """
    Test output 1
    """
    # 1. + 2.
    a = [
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4],
        [5, 5, 5, 5, 5],
    ]
    b = [
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4],
        [5, 5, 5, 5, 5],
    ]
    X = doe_al.dotdot_a_b_aT(a, b)
    # 3.
    assert np.allclose(
        X,
        np.array(
            [
                [75, 150, 225, 300, 375],
                [150, 300, 450, 600, 750],
                [225, 450, 675, 900, 1125],
                [300, 600, 900, 1200, 1500],
                [375, 750, 1125, 1500, 1875],
            ]
        ),
    )
