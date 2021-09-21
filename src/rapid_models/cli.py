"""Console script for rapid_models."""

import logging
import sys
import click
import click_log


logger = logging.getLogger(__name__)

click_log.basic_config(logger)


@click.command()
@click_log.simple_verbosity_option(logger)
def main(args=None):
    """Console script for rapid_models."""
    click.echo(
        "Replace this message by putting your code into " "rapid_models.cli.main"
    )
    click.echo("See click documentation at https://click.palletsprojects.com/")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
