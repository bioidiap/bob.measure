#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 21 Aug 2012 12:14:43 CEST
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

"""Script tests for bob.measure
"""

import os
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

def test_compute_perf():

  # sanity checks
  assert os.path.exists(DEV_SCORES)
  assert os.path.exists(TEST_SCORES)

  from .script.compute_perf import main
  cmdline = '--devel=%s --test=%s --self-test' % (DEV_SCORES, TEST_SCORES)
  nose.tools.eq_(main(cmdline.split()), 0)

def test_eval_threshold():

  # sanity checks
  assert os.path.exists(DEV_SCORES)

  from .script.eval_threshold import main
  cmdline = '--scores=%s --self-test' % (DEV_SCORES,)
  nose.tools.eq_(main(cmdline.split()), 0)

def test_apply_threshold():

  # sanity checks
  assert os.path.exists(TEST_SCORES)

  from .script.apply_threshold import main
  cmdline = '--scores=%s --self-test' % (TEST_SCORES,)
  nose.tools.eq_(main(cmdline.split()), 0)

def test_compute_perf_5col():

  # sanity checks
  assert os.path.exists(DEV_SCORES_5COL)
  assert os.path.exists(TEST_SCORES_5COL)

  from .script.compute_perf import main
  cmdline = '--devel=%s --test=%s --parser=bob.measure.load.split_five_column --self-test' % (DEV_SCORES_5COL, TEST_SCORES_5COL)
  nose.tools.eq_(main(cmdline.split()), 0)

def test_compute_cmc():

  # sanity checks
  assert os.path.exists(SCORES_4COL_CMC)
  assert os.path.exists(SCORES_5COL_CMC)

  from .script.plot_cmc import main
  nose.tools.eq_(main(['--self-test', '--score-file', SCORES_4COL_CMC, '--log-x-scale']), 0)
  nose.tools.eq_(main(['--self-test', '--score-file', SCORES_5COL_CMC, '--parser', '5column']), 0)
