# -*- coding: utf-8 -*-
"""Root level fixtures for `rapid_models`
"""
import pytest


@pytest.fixture
def your_fixture():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    return None


@pytest.fixture
def A_B_square_matrixes():
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
    return [a, b]
