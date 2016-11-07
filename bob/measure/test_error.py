#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed 11 Dec 15:14:08 2013 CET
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

"""Basic tests for the error measuring system of bob
"""

import os
import numpy
import nose.tools
import bob.io.base

def F(f):
  """Returns the test file on the "data" subdirectory"""
  import pkg_resources
  return pkg_resources.resource_filename(__name__, os.path.join('data', f))


def save(fname, data):
  """Saves a single array into a file in the 'data' directory."""
  bob.io.base.Array(data).save(os.path.join('data', fname))


def test_basic_ratios():

  from . import farfrr, precision_recall, f_score

  # We test the basic functionaly on FAR and FRR calculation. The first
  # example is separable, with a separation threshold of about 3.0

  positives = bob.io.base.load(F('linsep-positives.hdf5'))
  negatives = bob.io.base.load(F('linsep-negatives.hdf5'))

  minimum = min(positives.min(), negatives.min())
  maximum = max(positives.max(), negatives.max())

  # If we take a threshold on the minimum, the FAR should be 1.0 and the FRR
  # should be 0.0. Precision should be 0.5, recall should be 1.0
  far, frr = farfrr(negatives, positives, minimum-0.1)
  nose.tools.eq_(far, 1.0)
  nose.tools.eq_(frr, 0.0)
  prec, recall = precision_recall(negatives, positives, minimum-0.1)
  nose.tools.eq_(prec, 0.5)
  nose.tools.eq_(recall, 1.0)

  # Similarly, if we take a threshold on the maximum, the FRR should be 1.0
  # while the FAR should be 0.0. Both precision and recall should be 0.0.
  far, frr = farfrr(negatives, positives, maximum+0.1)
  nose.tools.eq_(far, 0.0)
  nose.tools.eq_(frr, 1.0)
  prec, recall = precision_recall(negatives, positives, maximum+0.1)
  nose.tools.eq_(prec, 0.0)
  nose.tools.eq_(recall, 0.0)

  # If we choose the appropriate threshold, we should get 0.0 for both FAR
  # and FRR. Precision will be 1.0, recall will be 1.0
  far, frr = farfrr(negatives, positives, 3.0)
  nose.tools.eq_(far, 0.0)
  nose.tools.eq_(frr, 0.0)
  prec, recall = precision_recall(negatives, positives, 3.0)
  nose.tools.eq_(prec, 1.0)
  nose.tools.eq_(recall, 1.0)

  # Testing the values of F-score depending on different choices of the threshold
  f_score_ = f_score(negatives, positives, minimum-0.1)
  nose.tools.assert_almost_equal(f_score_, 0.66666667)
  f_score_ = f_score(negatives, positives, minimum-0.1, 2)
  nose.tools.assert_almost_equal(f_score_, 0.83333333)

  f_score_ = f_score(negatives, positives, maximum+0.1)
  nose.tools.eq_(f_score_, 0.0)
  f_score_ = f_score(negatives, positives, maximum+0.1, 2)
  nose.tools.eq_(f_score_, 0.0)

  f_score_ = f_score(negatives, positives, 3.0)
  nose.tools.eq_(f_score_, 1.0)
  f_score_ = f_score(negatives, positives, 3.0, 2)
  nose.tools.eq_(f_score_, 1.0)


def test_indexing():

  from . import correctly_classified_positives, correctly_classified_negatives

  # This test verifies that the output of correctly_classified_positives() and
  # correctly_classified_negatives() makes sense.
  positives = bob.io.base.load(F('linsep-positives.hdf5'))
  negatives = bob.io.base.load(F('linsep-negatives.hdf5'))

  minimum = min(positives.min(), negatives.min())
  maximum = max(positives.max(), negatives.max())

  # If the threshold is minimum, we should have all positive samples
  # correctly classified and none of the negative samples correctly
  # classified.
  assert correctly_classified_positives(positives, minimum-0.1).all()
  assert not correctly_classified_negatives(negatives, minimum-0.1).any()

  # The inverse is true if the threshold is a bit above the maximum.
  assert not correctly_classified_positives(positives, maximum+0.1).any()
  assert correctly_classified_negatives(negatives, maximum+0.1).all()

  # If the threshold separates the sets, than all should be correctly
  # classified.
  assert correctly_classified_positives(positives, 3).all()
  assert correctly_classified_negatives(negatives, 3).all()


def test_thresholding():

  from . import eer_threshold, far_threshold, frr_threshold, farfrr, correctly_classified_positives, correctly_classified_negatives, min_hter_threshold

  def count(array, value=True):
    """Counts occurrences of a certain value in an array"""
    return list(array == value).count(True)

  # This example will demonstrate and check the use of eer_threshold() to
  # calculate the threshold that minimizes the EER.

  # This test set is not separable.
  positives = bob.io.base.load(F('nonsep-positives.hdf5'))
  negatives = bob.io.base.load(F('nonsep-negatives.hdf5'))
  threshold = eer_threshold(negatives, positives)

  sorted_positives = numpy.sort(positives)
  sorted_negatives = numpy.sort(negatives)

  # Of course we have to make sure that will set the EER correctly:
  ccp = count(correctly_classified_positives(positives,threshold))
  ccn = count(correctly_classified_negatives(negatives,threshold))
  assert (ccp - ccn) <= 1

  for t in (0, 0.001, 0.1, 0.5, 0.9, 0.999, 1):
    # Lets also test the far_threshold and the frr_threshold functions
    threshold_far = far_threshold(sorted_negatives, [], t, is_sorted=True)
    threshold_frr = frr_threshold([], sorted_positives, t, is_sorted=True)
    # Check that the requested FAR and FRR values are smaller than the requested ones
    far = farfrr(negatives, positives, threshold_far)[0]
    frr = farfrr(negatives, positives, threshold_frr)[1]
    assert far + 1e-7 > t
    assert frr + 1e-7 > t
    # test that the values are at least somewhere in the range
    assert far-t <= 0.15
    assert frr-t <= 0.15


  # If the set is separable, the calculation of the threshold is a little bit
  # trickier, as you have no points in the middle of the range to compare
  # things to. This is where the currently used recursive algorithm seems to
  # do better. Let's verify
  positives = bob.io.base.load(F('linsep-positives.hdf5'))
  negatives = bob.io.base.load(F('linsep-negatives.hdf5'))
  threshold = eer_threshold(negatives, positives)
  # the result here is 3.2 (which is what is expect ;-)
  assert threshold == 3.2

  # Of course we have to make sure that will set the EER correctly:
  ccp = count(correctly_classified_positives(positives,threshold))
  ccn = count(correctly_classified_negatives(negatives,threshold))
  nose.tools.eq_(ccp, ccn)

  # The second option for the calculation of the threshold is to use the
  # minimum HTER.
  threshold2 = min_hter_threshold(negatives, positives)
  assert threshold2 == 3.2
  nose.tools.eq_(threshold, threshold2) #in this particular case

  # Of course we have to make sure that will set the EER correctly:
  ccp = count(correctly_classified_positives(positives,threshold2))
  ccn = count(correctly_classified_negatives(negatives,threshold2))
  nose.tools.eq_(ccp, ccn)


def test_plots():

  from . import eer_threshold, roc, roc_for_far, precision_recall_curve, det, epc

  # This test set is not separable.
  positives = bob.io.base.load(F('nonsep-positives.hdf5'))
  negatives = bob.io.base.load(F('nonsep-negatives.hdf5'))
  threshold = eer_threshold(negatives, positives)

  # This example will test the ROC plot calculation functionality.
  xy = roc(negatives, positives, 100)
  # uncomment the next line to save a reference value
  # save('nonsep-roc.hdf5', xy)
  xyref = bob.io.base.load(F('nonsep-roc.hdf5'))
  assert numpy.array_equal(xy, xyref)

  # This example will test the ROC for FAR plot calculation functionality.
  far = [0.01, 0.1, 1]
  ref = [0.48, 0.22, 0]
  xy = roc_for_far(negatives, positives, far)
  # uncomment the next line to save a reference value
  assert numpy.array_equal(xy[0], far)
  assert numpy.array_equal(xy[1], ref)

  # This example will test the Precision-Recall plot calculation functionality.
  xy = precision_recall_curve(negatives, positives, 100)
  # uncomment the next line to save a reference value
  # save('nonsep-roc.hdf5', xy)
  xyref = bob.io.base.load(F('nonsep-precisionrecall.hdf5'))
  assert numpy.array_equal(xy, xyref)

  # This example will test the DET plot calculation functionality.
  det_xyzw = det(negatives, positives, 100)
  # uncomment the next line to save a reference value
  # save('nonsep-det.hdf5', det_xyzw)
  det_xyzw_ref = bob.io.base.load(F('nonsep-det.hdf5'))
  assert numpy.allclose(det_xyzw, det_xyzw_ref, atol=1e-15)

  # This example will test the EPC plot calculation functionality. For the
  # EPC curve, you need to have a development and a test set. We will split,
  # by the middle, the negatives and positives sample we have, just for the
  # sake of testing
  dev_negatives = negatives[:(negatives.shape[0]/2)]
  test_negatives = negatives[(negatives.shape[0]/2):]
  dev_positives = positives[:(positives.shape[0]/2)]
  test_positives = positives[(positives.shape[0]/2):]
  xy = epc(dev_negatives, dev_positives,
      test_negatives, test_positives, 100)
  # uncomment the next line to save a reference value
  # save('nonsep-epc.hdf5', xy)
  xyref = bob.io.base.load(F('nonsep-epc.hdf5'))
  assert numpy.allclose(xy, xyref, atol=1e-15)


def test_rocch():

  from . import rocch, rocch2eer, eer_rocch

  # This example will demonstrate and check the use of eer_rocch_threshold() to
  # calculate the threshold that minimizes the EER on the ROC Convex Hull

  # This test set is separable.
  positives = bob.io.base.load(F('linsep-positives.hdf5'))
  negatives = bob.io.base.load(F('linsep-negatives.hdf5'))
  # References obtained using Bosaris 1.06
  pmiss_pfa_ref = numpy.array([[1., 0., 0.], [0., 0., 1.]])
  eer_ref = 0.
  # Computes
  pmiss_pfa = rocch(negatives, positives)
  assert numpy.allclose(pmiss_pfa, pmiss_pfa_ref, atol=1e-15)
  eer = rocch2eer(pmiss_pfa)
  assert abs(eer-eer_ref) < 1e-4
  eer = eer_rocch(negatives, positives)
  assert abs(eer-eer_ref) < 1e-4

  # This test set is not separable.
  positives = bob.io.base.load(F('nonsep-positives.hdf5'))
  negatives = bob.io.base.load(F('nonsep-negatives.hdf5'))
  # References obtained using Bosaris 1.06
  pmiss_pfa_ref = numpy.array([[1., 0.68, 0.28, 0.1, 0.06, 0., 0.], [0, 0, 0.08, 0.12, 0.22, 0.48, 1.]])
  eer_ref = 0.116363636363636
  # Computes
  pmiss_pfa = rocch(negatives, positives)
  assert numpy.allclose(pmiss_pfa, pmiss_pfa_ref, atol=1e-15)
  eer = rocch2eer(pmiss_pfa)
  assert abs(eer-eer_ref) < 1e-4
  eer = eer_rocch(negatives, positives)
  assert abs(eer-eer_ref) < 1e-4


def test_cmc():

  from . import recognition_rate, cmc, load

  # tests the CMC calculation
  # test data; should give match characteristics [1/2,1/4,1/3] and CMC [1/3,2/3,1]
  test_data = [((0.3, 1.1, 0.5), (0.7,)), ((1.4, -1.3, 0.6), (0.2,)), ((0.8, 0., 1.5), (-0.8, 1.8)), ((2., 1.3, 1.6, 0.9), (2.4,))]
  # compute recognition rate
  rr = recognition_rate(test_data)
  nose.tools.eq_(rr, 0.5)
  # compute CMC
  cmc_ = cmc(test_data)
  assert (cmc_ == [0.5, 0.75, 1., 1., 1]).all()

  # load test data
  desired_rr = 0.76
  desired_cmc = [0.76, 0.89, 0.96, 0.98, 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
  data = load.cmc_four_column(F('scores-cmc-4col.txt'))
  rr = recognition_rate(data)
  nose.tools.eq_(rr, desired_rr)
  cmc_ = cmc(data)
  assert (cmc_ == desired_cmc).all()

  data = load.cmc_five_column(F('scores-cmc-5col.txt'))
  rr = recognition_rate(data)
  nose.tools.eq_(rr, desired_rr)
  cmc_ = cmc(data)
  assert (cmc_ == desired_cmc).all()


def test_calibration():

  from . import calibration

  # Tests the cllr and min_cllr measures
  # This test set is separable.
  positives = bob.io.base.load(F('linsep-positives.hdf5'))
  negatives = bob.io.base.load(F('linsep-negatives.hdf5'))

  cllr = calibration.cllr(negatives, positives)
  min_cllr = calibration.min_cllr(negatives, positives)

  assert min_cllr <= cllr
  nose.tools.assert_almost_equal(cllr, 1.2097942129)
  # Since the test set is separable, the min_cllr needs to be zero
  nose.tools.assert_almost_equal(min_cllr, 0.)

  # This test set is not separable.
  positives = bob.io.base.load(F('nonsep-positives.hdf5'))
  negatives = bob.io.base.load(F('nonsep-negatives.hdf5'))

  cllr = calibration.cllr(negatives, positives)
  min_cllr = calibration.min_cllr(negatives, positives)

  assert min_cllr <= cllr
  assert abs(cllr - 3.61833) < 1e-5, cllr
  assert abs(min_cllr - 0.33736) < 1e-5, min_cllr



def test_open_set_rates():

  # No error files
  cmc_scores = bob.measure.load.cmc_four_column(F("scores-cmc-4col-open-set.txt"))
  assert abs(bob.measure.detection_identification_rate(cmc_scores, threshold=0.5) - 1.0) < 1e-8
  assert abs(bob.measure.false_alarm_rate(cmc_scores, threshold=0.5)) < 1e-8

  assert abs(bob.measure.recognition_rate(cmc_scores) - 7./9.) < 1e-8
  assert abs(bob.measure.recognition_rate(cmc_scores, threshold=0.5) - 1.0) < 1e-8

  # One error
  cmc_scores = bob.measure.load.cmc_four_column(F("scores-cmc-4col-open-set-one-error.txt"))
  assert abs(bob.measure.detection_identification_rate(cmc_scores, threshold=0.5) - 6./7.) < 1e-8
  assert abs(bob.measure.false_alarm_rate(cmc_scores, threshold=0.5)) < 1e-8

  assert abs(bob.measure.recognition_rate(cmc_scores) - 6./9.) < 1e-8
  assert abs(bob.measure.recognition_rate(cmc_scores, threshold=0.5) - 6./7.) < 1e-8


  # Two errors
  cmc_scores = bob.measure.load.cmc_four_column(F("scores-cmc-4col-open-set-two-errors.txt"))
  assert abs(bob.measure.detection_identification_rate(cmc_scores, threshold=0.5) - 6./7.) < 1e-8
  assert abs(bob.measure.false_alarm_rate(cmc_scores, threshold=0.5) - 0.5) < 1e-8

  assert abs(bob.measure.recognition_rate(cmc_scores) - 6./9.) < 1e-8
  assert abs(bob.measure.recognition_rate(cmc_scores, threshold=0.5) - 6./8.) < 1e-8
