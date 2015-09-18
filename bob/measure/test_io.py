#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed 11 Dec 15:14:08 2013 CET
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

"""Tests the IO functionality of bob.measure."""

import bob.measure
import tempfile, os, shutil

import bob.io.base.test_utils

def test_load_scores():
  # This function tests the IO functionality of loading score files in different ways

  scores = []
  load_functions = {'4col' : bob.measure.load.four_column, '5col' : bob.measure.load.five_column}
  cols = {'4col' : 4, '5col' : 5}

  for variant in ('4col', '5col'):
    # read score file in normal way
    normal_score_file = bob.io.base.test_utils.datafile('dev-%s.txt' % variant, 'bob.measure')
    normal_scores = list(load_functions[variant](normal_score_file))

    assert len(normal_scores) == 910
    assert all(len(s) == cols[variant] for s in normal_scores)

    # read the compressed score file
    compressed_score_file = bob.io.base.test_utils.datafile('dev-%s.tar.gz' % variant, 'bob.measure')
    compressed_scores = list(load_functions[variant](compressed_score_file))

    assert len(compressed_scores) == len(normal_scores)
    assert all(len(c) == cols[variant] for c in compressed_scores)
    assert all(c[i] == s[i] for c,s in zip(compressed_scores, normal_scores) for i in range(cols[variant]))


def _check_binary_identical(name1, name2):
  # see: http://www.peterbe.com/plog/using-md5-to-check-equality-between-files
  from hashlib import md5
  # tests if two files are binary identical
  with open(name1,'rb') as f1:
    with open(name2,'rb') as f2:
      assert md5(f1.read()).digest() == md5(f2.read()).digest()


def test_openbr_verify():
  # This function tests that the conversion to the OpenBR verify file works as expected
  temp_dir = tempfile.mkdtemp(prefix='bob_test')

  # define output files
  openbr_extensions = ('.mtx', '.mask')
  matrix_file, mask_file = [os.path.join(temp_dir, "scores%s") % ext for ext in openbr_extensions]

  try:
    for variant in ('4col', '5col'):
      # get score file
      score_file = bob.io.base.test_utils.datafile('scores-cmc-%s.txt' % variant, 'bob.measure')

      # first round, do not define keyword arguments -- let the file get the gallery and probe ids automatically
      kwargs = {}
      for i in range(2):
        # get the files by automatically obtaining the identities
        bob.measure.openbr.write_matrix(score_file, matrix_file, mask_file, score_file_format = "%sumn" % variant, **kwargs)

        assert os.path.isfile(matrix_file) and os.path.isfile(mask_file)

        # check that they are binary identical to the reference files (which are tested to work and give the same results with OpenBR)
        matrix_ref, mask_ref = [bob.io.base.test_utils.datafile('scores%s' % ext, 'bob.measure') for ext in openbr_extensions]
        _check_binary_identical(matrix_file, matrix_ref)
        _check_binary_identical(mask_file, mask_ref)

        # define new kwargs for second round, i.e., define model and probe names
        # these names are identical to what is found in the score file, which in turn comes from the AT&T database
        model_type = {"4col" : "%d", "5col" : "s%d"}[variant]
        dev_ids = (3,4,7,8,9,13,15,18,19,22,23,25,28,30,31,32,35,37,38,40)
        kwargs['model_names'] = [model_type % c for c in dev_ids]
        kwargs['probe_names'] = ["s%d/%d" %(c,i) for c in dev_ids for i in (1,3,6,8,10)]

  finally:
    shutil.rmtree(temp_dir)


def test_openbr_search():
  # This function tests that the conversion to the OpenBR search file works as expected
  temp_dir = tempfile.mkdtemp(prefix='bob_test')

  # define output files
  openbr_extensions = ('.mtx', '.mask')
  matrix_file, mask_file = [os.path.join(temp_dir, "search%s") % ext for ext in openbr_extensions]

  try:
    for variant in ('4col', '5col'):
      # get score file
      score_file = bob.io.base.test_utils.datafile('scores-cmc-%s.txt' % variant, 'bob.measure')

      # first round, do not define keyword arguments -- let the file get the gallery and probe ids automatically
      kwargs = {}
      for i in range(2):
        # get the files by automatically obtaining the identities
        bob.measure.openbr.write_matrix(score_file, matrix_file, mask_file, score_file_format = "%sumn" % variant, search=50, **kwargs)

        assert os.path.isfile(matrix_file) and os.path.isfile(mask_file)

        # check that they are binary identical to the reference files (which are tested to work and give the same results with OpenBR)
        matrix_ref, mask_ref = [bob.io.base.test_utils.datafile('search%s' % ext, 'bob.measure') for ext in openbr_extensions]
        _check_binary_identical(matrix_file, matrix_ref)
        _check_binary_identical(mask_file, mask_ref)

        # define new kwargs for second round, i.e., define model and probe names
        # these names are identical to what is found in the score file, which in turn comes from the AT&T database
        model_type = {"4col" : "%d", "5col" : "s%d"}[variant]
        dev_ids = (3,4,7,8,9,13,15,18,19,22,23,25,28,30,31,32,35,37,38,40)
        kwargs['model_names'] = [model_type % c for c in dev_ids]
        kwargs['probe_names'] = ["s%d/%d" %(c,i) for c in dev_ids for i in (1,3,6,8,10)]

  finally:
    shutil.rmtree(temp_dir)
