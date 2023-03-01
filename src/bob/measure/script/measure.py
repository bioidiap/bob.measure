"""The main entry for bob.measure (click-based) scripts.
"""
import click
import pkg_resources

from clapper.click import AliasedGroup
from click_plugins import with_plugins


@with_plugins(pkg_resources.iter_entry_points("bob.measure.cli"))
@click.group(cls=AliasedGroup)
def measure():
    """Generic performance evaluation commands."""
    pass
