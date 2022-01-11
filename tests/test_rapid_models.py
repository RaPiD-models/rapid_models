#!/usr/bin/env python

"""Tests for `rapid_models` package."""

import pytest


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    return None


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    assert response is None
