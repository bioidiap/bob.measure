#!/usr/bin/env python
# coding=utf-8

"""Tests the IO functionality of bob.measure."""

import os

import h5py
import numpy
import pkg_resources

from bob.measure import load


def _F(f):
    """Returns the name of a file in the "data" subdirectory"""
    return pkg_resources.resource_filename(__name__, os.path.join("data", f))


def test_split():
    # This function test loading for generic bob.measure input files

    # Read test file
    test_file = _F("dev-1.txt")
    neg, pos = load.split(test_file)
    assert neg is not None
    assert len(neg) == 521
    assert len(pos) == 479

    with h5py.File(_F("two-cols.hdf5"), "r") as fh:
        neg_ref = fh["negatives"][:]
        pos_ref = fh["positives"][:]
    assert numpy.array_equal(numpy.nan_to_num(neg_ref), numpy.nan_to_num(neg))
    assert numpy.array_equal(numpy.nan_to_num(pos_ref), numpy.nan_to_num(pos))
