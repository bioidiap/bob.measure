'''Tests for bob.measure scripts'''

import sys
import filecmp
import click
from click.testing import CliRunner
import bob.io.base.test_utils
from .script import commands

def test_metrics():
    dev1 = bob.io.base.test_utils.datafile('dev-1.txt', 'bob.measure')
    runner = CliRunner()
    result = runner.invoke(commands.metrics, [dev1])
    with runner.isolated_filesystem():
        with open('tmp', 'w') as f:
            f.write(result.output)
        test_ref = bob.io.base.test_utils.datafile('test_m1.txt', 'bob.measure')
        assert result.exit_code == 0, (result.exit_code, result.output)

    dev2 = bob.io.base.test_utils.datafile('dev-2.txt', 'bob.measure')
    test1 = bob.io.base.test_utils.datafile('test-1.txt', 'bob.measure')
    test2 = bob.io.base.test_utils.datafile('test-2.txt', 'bob.measure')
    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['-e', dev1, test1, dev2, test2]
        )
        with open('tmp', 'w') as f:
            f.write(result.output)
        test_ref = bob.io.base.test_utils.datafile('test_m2.txt', 'bob.measure')
        assert result.exit_code == 0
    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['-e', '-l', 'tmp', dev1, test1, dev2, test2, '-lg',
                              'A,B']
        )
        assert result.exit_code == 0, (result.exit_code, result.output)

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['-l', 'tmp', dev1, dev2]
        )
        assert result.exit_code == 0, (result.exit_code, result.output)

def test_roc():
    dev1 = bob.io.base.test_utils.datafile('dev-1.txt', 'bob.measure')
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(commands.roc, ['--output', 'test.pdf',dev1])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0
    dev2 = bob.io.base.test_utils.datafile('dev-2.txt', 'bob.measure')
    test1 = bob.io.base.test_utils.datafile('test-1.txt', 'bob.measure')
    test2 = bob.io.base.test_utils.datafile('test-2.txt', 'bob.measure')
    with runner.isolated_filesystem():
        result = runner.invoke(commands.roc, ['--split', '--output',
                                              'test.pdf', '-e',
                                              '-ts', 'A,',
                                              dev1, test1, dev2, test2])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0, (result.exit_code, result.output)

    with runner.isolated_filesystem():
        result = runner.invoke(commands.roc, ['-e', '--output',
                                              'test.pdf', '--legends', 'A,B',
                                              dev1, test1, dev2, test2])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0, (result.exit_code, result.output)


def test_det():
    dev1 = bob.io.base.test_utils.datafile('dev-1.txt', 'bob.measure')
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(commands.det, [dev1])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0
    dev2 = bob.io.base.test_utils.datafile('dev-2.txt', 'bob.measure')
    test1 = bob.io.base.test_utils.datafile('test-1.txt', 'bob.measure')
    test2 = bob.io.base.test_utils.datafile('test-2.txt', 'bob.measure')
    with runner.isolated_filesystem():
        result = runner.invoke(commands.det, ['-e', '--split', '--output',
                                              'test.pdf', '--legends', 'A,B',
                                              '-lc', 'upper-right',
                                              dev1, test1, dev2, test2])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0, (result.exit_code, result.output)

    with runner.isolated_filesystem():
        result = runner.invoke(commands.det, ['--output',
                                              'test.pdf', '-e',
                                              dev1, test1, dev2, test2])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0, (result.exit_code, result.output)

def test_epc():
    dev1 = bob.io.base.test_utils.datafile('dev-1.txt', 'bob.measure')
    test1 = bob.io.base.test_utils.datafile('test-1.txt', 'bob.measure')
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(commands.epc, [dev1, test1])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0, (result.exit_code, result.output)

    dev2 = bob.io.base.test_utils.datafile('dev-2.txt', 'bob.measure')
    test2 = bob.io.base.test_utils.datafile('test-2.txt', 'bob.measure')
    with runner.isolated_filesystem():
        result = runner.invoke(commands.epc, ['--output', 'test.pdf',
                                              '--legends', 'A,B',
                                              '--titles', 'TA,TB',
                                              '-lc', 'upper-right',
                                              dev1, test1, dev2, test2])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0, (result.exit_code, result.output)

def test_hist():
    dev1 = bob.io.base.test_utils.datafile('dev-1.txt', 'bob.measure')
    test1 = bob.io.base.test_utils.datafile('test-1.txt', 'bob.measure')
    dev2 = bob.io.base.test_utils.datafile('dev-2.txt', 'bob.measure')
    test2 = bob.io.base.test_utils.datafile('test-2.txt', 'bob.measure')
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(commands.hist, [dev1])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0, (result.exit_code, result.output, result.exception)

    with runner.isolated_filesystem():
        result = runner.invoke(commands.hist, ['--criterion',
                                               'min-hter', '--no-line',
                                               '--output', 'HISTO.pdf',  '-b',
                                               '30,100,30,100', dev1, dev2])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0, (result.exit_code, result.output)

    with runner.isolated_filesystem():
        result = runner.invoke(commands.hist, ['-e', '--criterion', 'eer','--output',
                                               'HISTO.pdf',  '-b',  '30,20,30,20',
                                               '-sp', 221, '-lg', 'A,B',
                                               dev1, test1, dev2, test2])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0, (result.exit_code, result.output)


def test_evaluate():
    dev1 = bob.io.base.test_utils.datafile('dev-1.txt', 'bob.measure')
    test1 = bob.io.base.test_utils.datafile('test-1.txt', 'bob.measure')
    dev2 = bob.io.base.test_utils.datafile('dev-2.txt', 'bob.measure')
    test2 = bob.io.base.test_utils.datafile('test-2.txt', 'bob.measure')
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(commands.evaluate, [dev1])
        assert result.exit_code == 0, (result.exit_code, result.output, result.exception)

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.evaluate, ['--output', 'my_plots.pdf',
                                '-n', 300, dev1, dev2])
        assert result.exit_code == 0, (result.exit_code, result.output)

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.evaluate, ['-e', dev1, test1, dev2, test2])
        assert result.exit_code == 0, (result.exit_code, result.output)
