"""The main entry for bob.measure (click-based) scripts.
"""
import click
import pkg_resources
from click_plugins import with_plugins


@with_plugins(pkg_resources.iter_entry_points('bob.measure.cli'))
@click.group()
def measure():
    """Entry for bob.measure commands."""
    pass
