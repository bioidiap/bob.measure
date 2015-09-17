#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed 11 Dec 15:14:08 2013 CET
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

"""Tests the IO functionality of bob.measure."""

import bob.measure
import pkg_resources

def test_load_scores():
  # This function tests the IO functionality of loading score files in different ways

  scores = []
  load_functions = {'4col' : bob.measure.load.four_column, '5col' : bob.measure.load.five_column}
  cols = {'4col' : 4, '5col' : 5}

  for variant in ('4col', '5col'):

    # read score file in normal way
    normal_score_file = pkg_resources.resource_filename('bob.measure', 'data/dev-%s.txt' % variant)
    normal_scores = list(load_functions[variant](normal_score_file))

    assert len(normal_scores) == 910
    assert all(len(s) == cols[variant] for s in normal_scores)

    # read the compressed score file
    compressed_score_file = pkg_resources.resource_filename('bob.measure', 'data/dev-%s.tar.gz' % variant)
    compressed_scores = list(load_functions[variant](compressed_score_file))

    assert len(compressed_scores) == len(normal_scores)
    assert all(len(c) == cols[variant] for c in compressed_scores)
    assert all(c[i] == s[i] for c,s in zip(compressed_scores, normal_scores) for i in range(cols[variant]))
