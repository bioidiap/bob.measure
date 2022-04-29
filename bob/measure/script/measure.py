"""The main entry for bob.measure (click-based) scripts.
"""
import click
import pkg_resources

from click_plugins import with_plugins

from bob.extension.scripts.click_helper import AliasedGroup


@with_plugins(pkg_resources.iter_entry_points("bob.measure.cli"))
@click.group(cls=AliasedGroup)
def measure():
    """Generic performance evaluation commands."""
    pass
