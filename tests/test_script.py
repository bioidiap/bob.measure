"""Tests for bob.measure scripts"""

import os

import click
import pkg_resources

from click.testing import CliRunner

from bob.io.base.testing_utils import assert_click_runner_result
from bob.measure.script import commands


def _F(f):
    """Returns the name of a file in the "data" subdirectory"""
    return pkg_resources.resource_filename(__name__, os.path.join("data", f))


def test_metrics():
    dev1 = _F("dev-1.txt")
    runner = CliRunner()
    result = runner.invoke(commands.metrics, [dev1])
    with runner.isolated_filesystem():
        with open("tmp", "w") as f:
            f.write(result.output)
        assert_click_runner_result(result)

    dev2 = _F("dev-2.txt")
    test1 = _F("test-1.txt")
    test2 = _F("test-2.txt")
    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ["-e", dev1, test1, dev2, test2]
        )
        with open("tmp", "w") as f:
            f.write(result.output)
        assert_click_runner_result(result)
    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics,
            ["-e", "-l", "tmp", dev1, test1, dev2, test2, "-lg", "A,B"],
        )
        assert_click_runner_result(result)

    with runner.isolated_filesystem():
        result = runner.invoke(commands.metrics, ["-l", "tmp", dev1, dev2])
        assert_click_runner_result(result)


def test_roc():
    dev1 = _F("dev-1.txt")
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(commands.roc, ["--output", "test.pdf", dev1])
        if result.output:
            click.echo(result.output)
        assert_click_runner_result(result)
    dev2 = _F("dev-2.txt")
    test1 = _F("test-1.txt")
    test2 = _F("test-2.txt")
    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.roc,
            [
                "--split",
                "--output",
                "test.pdf",
                "-e",
                "-ts",
                "A,",
                dev1,
                test1,
                dev2,
                test2,
            ],
        )
        if result.output:
            click.echo(result.output)
        assert_click_runner_result(result)

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.roc,
            [
                "-e",
                "--output",
                "test.pdf",
                "--legends",
                "A,B",
                dev1,
                test1,
                dev2,
                test2,
            ],
        )
        if result.output:
            click.echo(result.output)
        assert_click_runner_result(result)


def test_det():
    dev1 = _F("dev-1.txt")
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(commands.det, [dev1])
        if result.output:
            click.echo(result.output)
        assert_click_runner_result(result)
    dev2 = _F("dev-2.txt")
    test1 = _F("test-1.txt")
    test2 = _F("test-2.txt")
    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.det,
            [
                "-e",
                "--split",
                "--output",
                "test.pdf",
                "--legends",
                "A,B",
                "-ll",
                "upper-right",
                dev1,
                test1,
                dev2,
                test2,
            ],
        )
        if result.output:
            click.echo(result.output)
        assert_click_runner_result(result)

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.det,
            ["--output", "test.pdf", "-e", dev1, test1, dev2, test2],
        )
        if result.output:
            click.echo(result.output)
        assert_click_runner_result(result)


def test_epc():
    dev1 = _F("dev-1.txt")
    test1 = _F("test-1.txt")
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(commands.epc, [dev1, test1])
        if result.output:
            click.echo(result.output)
        assert_click_runner_result(result)

    dev2 = _F("dev-2.txt")
    test2 = _F("test-2.txt")
    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.epc,
            [
                "--output",
                "test.pdf",
                "--legends",
                "A,B",
                "--titles",
                "TA,TB",
                "-ll",
                "upper-right",
                dev1,
                test1,
                dev2,
                test2,
            ],
        )
        if result.output:
            click.echo(result.output)
        assert_click_runner_result(result)


def test_hist():
    dev1 = _F("dev-1.txt")
    test1 = _F("test-1.txt")
    dev2 = _F("dev-2.txt")
    test2 = _F("test-2.txt")
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(commands.hist, [dev1])
        if result.output:
            click.echo(result.output)
        assert_click_runner_result(result)

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.hist,
            [
                "--criterion",
                "min-hter",
                "--no-line",
                "--output",
                "HISTO.pdf",
                "-b",
                "30,100",
                dev1,
                dev2,
            ],
        )
        if result.output:
            click.echo(result.output)
        assert_click_runner_result(result)

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.hist,
            [
                "-e",
                "--criterion",
                "eer",
                "--output",
                "HISTO.pdf",
                "-b",
                "30,20",
                "-sp",
                111,
                "-ts",
                "A,B",
                dev1,
                test1,
                dev2,
                test2,
            ],
        )
        if result.output:
            click.echo(result.output)
        assert_click_runner_result(result)


def test_hist_legends():
    dev1 = _F("dev-1.txt")
    test1 = _F("test-1.txt")
    dev2 = _F("dev-2.txt")
    test2 = _F("test-2.txt")
    runner = CliRunner()

    # share same title for dev/eval of each system
    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.hist,
            ["-e", "-sp", 111, "-ts", "A,B", dev1, test1, dev2, test2],
        )
        if result.output:
            click.echo(result.output)
        assert_click_runner_result(result)

    # individual titles for dev and eval
    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.hist,
            ["-e", "-sp", 121, "-ts", "A,B,C,D", dev1, test1, dev2, test2],
        )
        if result.output:
            click.echo(result.output)
        assert_click_runner_result(result)


def test_evaluate():
    dev1 = _F("dev-1.txt")
    test1 = _F("test-1.txt")
    dev2 = _F("dev-2.txt")
    test2 = _F("test-2.txt")
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(commands.evaluate, [dev1])
        assert_click_runner_result(result)

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.evaluate,
            ["--output", "my_plots.pdf", "-n", 300, dev1, dev2],
        )
        assert_click_runner_result(result)

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.evaluate, ["-e", dev1, test1, dev2, test2]
        )
        assert_click_runner_result(result)
