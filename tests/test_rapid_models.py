#!/usr/bin/env python

"""Tests for `rapid_models` package."""

import pytest

from click.testing import CliRunner

from rapid_models import rapid_models
from rapid_models import cli


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    return None


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    assert response is None


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert 'rapid_models.cli.main' in result.output
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert '--help' in help_result.output and 'Show this message and exit' in help_result.output
