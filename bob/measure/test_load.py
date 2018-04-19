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
import bob.io.base

import numpy


def test_split():
    # This function test loading for generic bob.measure input files

    # Read test file
    test_file = bob.io.base.test_utils.datafile(
        'dev-1.txt', 'bob.measure')
    neg, pos = bob.measure.load.split(test_file)
    assert neg is not None
    assert_equal(len(neg), 521)
    assert_equal(len(pos), 479)

    test_ref_file_path = bob.io.base.test_utils.datafile(
        'two-cols.hdf5', 'bob.measure')
    test_ref = bob.io.base.HDF5File(test_ref_file_path)
    neg_ref = test_ref.read('negatives')
    pos_ref = test_ref.read('positives')
    del test_ref
    assert numpy.array_equal(numpy.nan_to_num(neg_ref), numpy.nan_to_num(neg))
    assert numpy.array_equal(numpy.nan_to_num(pos_ref), numpy.nan_to_num(pos))
