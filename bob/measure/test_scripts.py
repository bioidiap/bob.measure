#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tue 21 Aug 2012 12:14:43 CEST

"""Script tests for bob.measure
"""

import os
import tempfile

import nose.tools
import pkg_resources


def F(f):
  """Returns the test file on the "data" subdirectory"""
  return pkg_resources.resource_filename(__name__, os.path.join('data', f))

DEV_SCORES = F('dev-4col.txt')
TEST_SCORES = F('test-4col.txt')

DEV_SCORES_5COL = F('dev-5col.txt')
TEST_SCORES_5COL = F('test-5col.txt')

SCORES_4COL_CMC = F('scores-cmc-4col.txt')
SCORES_5COL_CMC = F('scores-cmc-5col.txt')

SCORES_4COL_CMC_OS = F('scores-cmc-4col-open-set.txt')


def test_compute_perf():

  # sanity checks
  assert os.path.exists(DEV_SCORES)
  assert os.path.exists(TEST_SCORES)

  tmp_output = tempfile.NamedTemporaryFile(prefix=__name__, suffix='.pdf')

  cmdline = [
      DEV_SCORES,
      TEST_SCORES,
      '--output=' + tmp_output.name,
      ]

  from .script.compute_perf import main
  nose.tools.eq_(main(cmdline), 0)


def test_eval_threshold():

  # sanity checks
  assert os.path.exists(DEV_SCORES)

  cmdline = [DEV_SCORES]

  from .script.eval_threshold import main
  nose.tools.eq_(main(cmdline), 0)


def test_apply_threshold():

  # sanity checks
  assert os.path.exists(TEST_SCORES)

  cmdline = [
      '0.5',
      TEST_SCORES,
      ]

  from .script.apply_threshold import main
  nose.tools.eq_(main(cmdline), 0)


def test_compute_perf_5col():

  # sanity checks
  assert os.path.exists(DEV_SCORES_5COL)
  assert os.path.exists(TEST_SCORES_5COL)

  tmp_output = tempfile.NamedTemporaryFile(prefix=__name__, suffix='.pdf')

  cmdline = [
      DEV_SCORES_5COL,
      TEST_SCORES_5COL,
      '--output=' + tmp_output.name,
      ]

  from .script.compute_perf import main
  nose.tools.eq_(main(cmdline), 0)


def test_compute_cmc():

  # sanity checks
  assert os.path.exists(SCORES_4COL_CMC)
  assert os.path.exists(SCORES_5COL_CMC)
  assert os.path.exists(SCORES_4COL_CMC_OS)

  from .script.plot_cmc import main

  tmp_output = tempfile.NamedTemporaryFile(prefix=__name__, suffix='.pdf')

  nose.tools.eq_(main([
    SCORES_4COL_CMC,
    '--log-x-scale',
    '--output=%s' % tmp_output.name,
    ]), 0)

  tmp_output = tempfile.NamedTemporaryFile(prefix=__name__, suffix='.pdf')

  nose.tools.eq_(main([
    SCORES_5COL_CMC,
    '--output=%s' % tmp_output.name,
    ]), 0)

  tmp_output = tempfile.NamedTemporaryFile(prefix=__name__, suffix='.pdf')

  nose.tools.eq_(main([
    SCORES_4COL_CMC_OS,
    '--rank=1',
    '--output=%s' % tmp_output.name,
    ]), 0)
