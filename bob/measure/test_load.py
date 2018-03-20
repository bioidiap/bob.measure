#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed 11 Dec 15:14:08 2013 CET
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

"""Tests the IO functionality of bob.measure."""

from nose.tools import assert_equal
import bob.measure.load
import bob.io.base.test_utils


def test_split():
    # This function test loading for generic bob.measure input files

    # Read test file
    test_file = bob.io.base.test_utils.datafile(
        'data.txt', 'bob.measure')
    neg, pos = bob.measure.load.split(test_file)
    assert neg is not None
    assert_equal(len(neg), 521)
    assert_equal(len(pos), 479)
