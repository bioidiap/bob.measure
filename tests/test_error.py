#!/usr/bin/env python
# coding=utf-8

"""Basic tests for the error measuring functions
"""

import math
import os

import h5py
import numpy
import pkg_resources
import pytest

from bob.measure import (
    calibration,
    cmc,
    correctly_classified_negatives,
    correctly_classified_positives,
    det,
    detection_identification_rate,
    eer_rocch,
    eer_threshold,
    epc,
    f_score,
    false_alarm_rate,
    far_threshold,
    farfrr,
    frr_threshold,
    min_hter_threshold,
    min_weighted_error_rate_threshold,
    precision_recall,
    precision_recall_curve,
    recognition_rate,
    roc,
    roc_auc_score,
    roc_for_far,
    rocch,
    rocch2eer,
)


def _F(f):
    """Returns the name of a file in the "data" subdirectory"""
    return pkg_resources.resource_filename(__name__, os.path.join("data", f))


def _load(f):
    """Loads an array contained in a file from the "data" subdirectory."""
    with h5py.File(_F(f), "r") as fh:
        return fh["array"][:]


def _load_many(f, datasets):
    """Loads an array contained in a file from the "data" subdirectory."""
    with h5py.File(_F(f), "r") as fh:
        return [fh[k][:] for k in datasets]


def _save(f, data):
    """Saves a single array into a file in the "data" directory."""
    with h5py.File(_F(f), "w") as fh:
        fh.create_dataset(
            "array", data=data, compression="gzip", compression_opts=9
        )


def test_basic_ratios():
    # We test the basic functionaly on FAR and FRR calculation. The first
    # example is separable, with a separation threshold of about 3.0

    positives = _load("linsep-positives.hdf5")
    negatives = _load("linsep-negatives.hdf5")

    minimum = min(positives.min(), negatives.min())
    maximum = max(positives.max(), negatives.max())

    # If we take a threshold on the minimum, the FAR should be 1.0 and the FRR
    # should be 0.0. Precision should be 0.5, recall should be 1.0
    far, frr = farfrr(negatives, positives, minimum - 0.1)
    assert far == 1.0
    assert frr == 0.0
    prec, recall = precision_recall(negatives, positives, minimum - 0.1)
    assert prec == 0.5
    assert recall == 1.0

    # Similarly, if we take a threshold on the maximum, the FRR should be 1.0
    # while the FAR should be 0.0. Both precision and recall should be 0.0.
    far, frr = farfrr(negatives, positives, maximum + 0.1)
    assert far == 0.0
    assert frr == 1.0
    prec, recall = precision_recall(negatives, positives, maximum + 0.1)
    assert prec == 0.0
    assert recall == 0.0

    # If we choose the appropriate threshold, we should get 0.0 for both FAR
    # and FRR. Precision will be 1.0, recall will be 1.0
    far, frr = farfrr(negatives, positives, 3.0)
    assert far == 0.0
    assert frr == 0.0
    prec, recall = precision_recall(negatives, positives, 3.0)
    assert prec == 1.0
    assert recall == 1.0

    # Testing the values of F-score depending on different choices of the
    # threshold
    f_score_ = f_score(negatives, positives, minimum - 0.1)
    assert numpy.isclose(f_score_, 0.66666667)
    f_score_ = f_score(negatives, positives, minimum - 0.1, 2)
    assert numpy.isclose(f_score_, 0.83333333)

    f_score_ = f_score(negatives, positives, maximum + 0.1)
    assert f_score_ == 0.0
    f_score_ = f_score(negatives, positives, maximum + 0.1, 2)
    assert f_score_ == 0.0

    f_score_ = f_score(negatives, positives, 3.0)
    assert f_score_ == 1.0
    f_score_ = f_score(negatives, positives, 3.0, 2)
    assert f_score_ == 1.0


def test_for_uncomputable_thresholds():
    # in some cases, we cannot compute an FAR or FRR threshold, e.g., when we
    # have too little data or too many equal scores in these cases, the methods
    # should return a threshold which a supports a lower value.

    # case 1: several scores are identical
    pos = [0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    neg = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0]

    # test that reasonable thresholds for reachable data points are provided
    threshold = far_threshold(neg, pos, 0.5)
    assert threshold == 1.0, threshold
    threshold = frr_threshold(neg, pos, 0.5)
    assert numpy.isclose(threshold, 0.1)

    threshold = far_threshold(neg, pos, 0.4)
    assert threshold > neg[-1]
    threshold = frr_threshold(neg, pos, 0.4)
    assert threshold >= pos[0]

    # test the same with even number of scores
    pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    neg = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0]

    threshold = far_threshold(neg, pos, 0.5)
    assert threshold == 1.0
    assert numpy.isclose(frr_threshold(neg, pos, 0.51), 0.1)
    threshold = far_threshold(neg, pos, 0.49)
    assert threshold > neg[-1]
    threshold = frr_threshold(neg, pos, 0.49)
    assert threshold >= pos[0]

    # case 2: too few scores for the desired threshold
    pos = numpy.array(range(10), dtype=float)
    neg = numpy.array(range(10), dtype=float)

    threshold = far_threshold(neg, pos, 0.09)
    assert threshold > neg[-1]
    threshold = frr_threshold(neg, pos, 0.09)
    assert threshold >= pos[0]
    # there is no limit above; the threshold will just be the largest possible
    # value
    threshold = far_threshold(neg, pos, 0.11)
    assert threshold == 9.0
    threshold = far_threshold(neg, pos, 0.91)
    assert threshold == 1.0
    threshold = far_threshold(neg, pos, 1)
    assert threshold <= 0.0
    threshold = frr_threshold(neg, pos, 0.11)
    assert numpy.isclose(threshold, 1.0)
    threshold = frr_threshold(neg, pos, 0.91)
    assert numpy.isclose(threshold, 9.0)


def test_indexing():
    # This test verifies that the output of correctly_classified_positives() and
    # correctly_classified_negatives() makes sense.
    positives = _load("linsep-positives.hdf5")
    negatives = _load("linsep-negatives.hdf5")

    minimum = min(positives.min(), negatives.min())
    maximum = max(positives.max(), negatives.max())

    # If the threshold is minimum, we should have all positive samples
    # correctly classified and none of the negative samples correctly
    # classified.
    assert correctly_classified_positives(positives, minimum - 0.1).all()
    assert not correctly_classified_negatives(negatives, minimum - 0.1).any()

    # The inverse is true if the threshold is a bit above the maximum.
    assert not correctly_classified_positives(positives, maximum + 0.1).any()
    assert correctly_classified_negatives(negatives, maximum + 0.1).all()

    # If the threshold separates the sets, than all should be correctly
    # classified.
    assert correctly_classified_positives(positives, 3).all()
    assert correctly_classified_negatives(negatives, 3).all()


def test_obvious_thresholds():
    M = 10
    neg = numpy.arange(M, dtype=float)
    pos = numpy.arange(M, 2 * M, dtype=float)

    for far, frr in zip(
        numpy.arange(0, 2 * M + 1, dtype=float) / M / 2,
        numpy.arange(0, 2 * M + 1, dtype=float) / M / 2,
    ):
        far, expected_far = round(far, 2), math.floor(far * 10) / 10
        frr, expected_frr = round(frr, 2), math.floor(frr * 10) / 10
        calculated_far_threshold = far_threshold(neg, pos, far)
        pred_far, _ = farfrr(neg, pos, calculated_far_threshold)

        calculated_frr_threshold = frr_threshold(neg, pos, frr)
        _, pred_frr = farfrr(neg, pos, calculated_frr_threshold)
        assert pred_far <= far
        assert pred_far == expected_far
        assert pred_frr <= frr
        assert pred_frr == expected_frr


def test_thresholding():
    # This example will demonstrate and check the use of eer_threshold() to
    # calculate the threshold that minimizes the EER.

    # This test set is not separable.
    positives = _load("nonsep-positives.hdf5")
    negatives = _load("nonsep-negatives.hdf5")

    sorted_negatives = numpy.sort(negatives)
    sorted_positives = numpy.sort(positives)
    threshold = eer_threshold(
        sorted_negatives, sorted_positives, is_sorted=True
    )

    # Of course we have to make sure that will set the EER correctly:
    ccp = correctly_classified_positives(positives, threshold).sum()
    ccn = correctly_classified_negatives(negatives, threshold).sum()
    assert (ccp - ccn) <= 1

    for t in (0, 0.001, 0.1, 0.5, 0.9, 0.999, 1):
        # Lets also test the far_threshold and the frr_threshold functions
        threshold_far = far_threshold(sorted_negatives, [], t, is_sorted=True)
        threshold_frr = frr_threshold([], sorted_positives, t, is_sorted=True)
        # Check that the requested FAR and FRR values are smaller than the
        # requested ones
        far = farfrr(negatives, positives, threshold_far)[0]
        frr = farfrr(negatives, positives, threshold_frr)[1]
        if not math.isnan(threshold_far):
            assert far <= t
            assert t - far <= 0.1
        if not math.isnan(threshold_frr):
            assert frr <= t
            # test that the values are at least somewhere in the range
            assert t - frr <= 0.1

    # If the set is separable, the calculation of the threshold is a little bit
    # trickier, as you have no points in the middle of the range to compare
    # things to. This is where the currently used recursive algorithm seems to
    # do better. Let's verify
    positives = _load("linsep-positives.hdf5")
    negatives = _load("linsep-negatives.hdf5")
    threshold = eer_threshold(negatives, positives)
    # the result here is 3.2 (which is what is expect ;-)
    assert threshold == 3.2

    # Of course we have to make sure that will set the EER correctly:
    ccp = correctly_classified_positives(positives, threshold).sum()
    ccn = correctly_classified_negatives(negatives, threshold).sum()
    assert ccp == ccn

    # The second option for the calculation of the threshold is to use the
    # minimum HTER.
    threshold2 = min_hter_threshold(negatives, positives)
    assert threshold2 == 3.2
    assert threshold == threshold2  # in this particular case

    # Of course we have to make sure that will set the EER correctly:
    ccp = correctly_classified_positives(positives, threshold2).sum()
    ccn = correctly_classified_negatives(negatives, threshold2).sum()
    assert ccp == ccn


def test_empty_raises():
    for func in (
        farfrr,
        precision_recall,
        f_score,
        min_weighted_error_rate_threshold,
    ):
        with pytest.raises(RuntimeError):
            func([], [1.0], 0)

        with pytest.raises(RuntimeError):
            func([1.0], [], 0)

        with pytest.raises(RuntimeError):
            func([], [], 0)

    for func in (eer_threshold, min_hter_threshold):
        with pytest.raises(RuntimeError):
            func(negatives=[], positives=[1.0])

        with pytest.raises(RuntimeError):
            func(negatives=[1.0], positives=[])

        with pytest.raises(RuntimeError):
            func(negatives=[], positives=[])


def test_plots():
    # This test set is not separable.
    positives = _load("nonsep-positives.hdf5")
    negatives = _load("nonsep-negatives.hdf5")
    eer_threshold(negatives, positives)

    # This example will test the ROC plot calculation functionality.
    xy = roc(negatives, positives, 100)
    # uncomment the next line to save a reference value
    # _save("nonsep-roc.hdf5", xy)
    xyref = _load("nonsep-roc.hdf5")
    numpy.testing.assert_array_equal(xy, xyref)

    # This example will test the ROC for FAR plot calculation functionality.
    requested_far = [0.01, 0.1, 1]
    expected_far = [0.0, 0.1, 1]
    expected_frr = [0.48, 0.12, 0]
    xy = roc_for_far(negatives, positives, requested_far)

    numpy.testing.assert_array_equal(xy[0], expected_far)
    numpy.testing.assert_array_equal(xy[1], expected_frr)

    # This example will test the Precision-Recall plot calculation functionality.
    xy = precision_recall_curve(negatives, positives, 100)
    # uncomment the next line to save a reference value
    # _save("nonsep-precisionrecall.hdf5", xy)
    xyref = _load("nonsep-precisionrecall.hdf5")
    numpy.testing.assert_array_equal(xy, xyref)

    # This example will test the DET plot calculation functionality.
    det_xyzw = det(negatives, positives, 100)
    # uncomment the next line to save a reference value
    # _save("nonsep-det.hdf5", det_xyzw)
    det_xyzw_ref = _load("nonsep-det.hdf5")
    numpy.testing.assert_allclose(det_xyzw, det_xyzw_ref, atol=1e-5, rtol=1e-6)

    # This example will test the EPC plot calculation functionality. For the
    # EPC curve, you need to have a development and a test set. We will split,
    # by the middle, the negatives and positives sample we have, just for the
    # sake of testing
    dev_negatives = negatives[: (negatives.shape[0] // 2)]
    test_negatives = negatives[(negatives.shape[0] // 2) :]
    dev_positives = positives[: (positives.shape[0] // 2)]
    test_positives = positives[(positives.shape[0] // 2) :]
    xy = epc(dev_negatives, dev_positives, test_negatives, test_positives, 100)
    xyref = _load("nonsep-epc.hdf5")
    numpy.testing.assert_allclose(xy[0], xyref[0], atol=1e-5, rtol=1e-6)
    numpy.testing.assert_allclose(xy[1], xyref[1], atol=1e-5, rtol=1e-6)
    xy = epc(
        dev_negatives,
        dev_positives,
        test_negatives,
        test_positives,
        100,
        False,
        True,
    )
    # uncomment the next line to save a reference value
    # _save("nonsep-epc.hdf5", xy)
    numpy.testing.assert_allclose(xy[2], xyref[2], atol=1e-5, rtol=1e-6)
    numpy.testing.assert_allclose(xy[0], xyref[0], atol=1e-5, rtol=1e-6)
    numpy.testing.assert_allclose(xy[1], xyref[1], atol=1e-5, rtol=1e-6)


def no_test_rocch():
    # This example will demonstrate and check the use of eer_rocch_threshold() to
    # calculate the threshold that minimizes the EER on the ROC Convex Hull

    # This test set is separable.
    positives = _load("linsep-positives.hdf5")
    negatives = _load("linsep-negatives.hdf5")
    # References obtained using Bosaris 1.06
    pmiss_pfa_ref = numpy.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    eer_ref = 0.0
    # Computes
    pmiss_pfa = rocch(negatives, positives)
    numpy.testing.assert_allclose(pmiss_pfa, pmiss_pfa_ref, atol=1e-15)
    eer = rocch2eer(pmiss_pfa)
    assert abs(eer - eer_ref) < 1e-4
    eer = eer_rocch(negatives, positives)
    assert abs(eer - eer_ref) < 1e-4

    # This test set is not separable.
    positives = _load("nonsep-positives.hdf5")
    negatives = _load("nonsep-negatives.hdf5")
    # References obtained using Bosaris 1.06
    pmiss_pfa_ref = numpy.array(
        [
            [1.0, 0.68, 0.28, 0.1, 0.06, 0.0, 0.0],
            [0, 0, 0.08, 0.12, 0.22, 0.48, 1.0],
        ]
    )
    eer_ref = 0.116363636363636
    # Computes
    pmiss_pfa = rocch(negatives, positives)
    numpy.testing.assert_allclose(pmiss_pfa, pmiss_pfa_ref, atol=1e-15)
    eer = rocch2eer(pmiss_pfa)
    assert abs(eer - eer_ref) < 1e-4
    eer = eer_rocch(negatives, positives)
    assert abs(eer - eer_ref) < 1e-4


def test_cmc():
    # tests the CMC calculation
    # test data; should give match characteristics [1/2,1/4,1/3] and CMC
    # [1/3,2/3,1]
    test_data = [
        ((0.3, 1.1, 0.5), (0.7,)),
        ((1.4, -1.3, 0.6), (0.2,)),
        ((0.8, 0.0, 1.5), (-0.8, 1.8)),
        ((2.0, 1.3, 1.6, 0.9), (2.4,)),
    ]
    # compute recognition rate
    rr = recognition_rate(test_data)
    assert rr == 0.5
    # compute CMC
    cmc_ = cmc(test_data)
    assert (cmc_ == [0.5, 0.75, 1.0, 1.0, 1]).all()

    # load test data
    desired_rr = 0.76
    desired_cmc = [
        0.76,
        0.89,
        0.96,
        0.98,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    data = _load_many("test-cmc.hdf5", ("data-neg", "data-pos"))
    data = list(zip(data[0], data[1]))
    rr = recognition_rate(data)
    assert rr == desired_rr
    cmc_ = cmc(data)
    assert (cmc_ == desired_cmc).all()


def no_test_calibration():
    # Tests the cllr and min_cllr measures
    # This test set is separable.
    positives = _load("linsep-positives.hdf5")
    negatives = _load("linsep-negatives.hdf5")

    cllr = calibration.cllr(negatives, positives)
    min_cllr = calibration.min_cllr(negatives, positives)

    assert min_cllr <= cllr
    numpy.isclose(cllr, 1.2097942129)
    # Since the test set is separable, the min_cllr needs to be zero
    numpy.isclose(min_cllr, 0.0)

    # This test set is not separable.
    positives = _load("nonsep-positives.hdf5")
    negatives = _load("nonsep-negatives.hdf5")

    cllr = calibration.cllr(negatives, positives)
    min_cllr = calibration.min_cllr(negatives, positives)

    assert min_cllr <= cllr
    assert abs(cllr - 3.61833) < 1e-5
    assert abs(min_cllr - 0.33736) < 1e-5


def test_open_set_rates():
    def _load_cmc_scores(fname):
        with h5py.File(_F(fname), "r") as fh:
            negative = []
            positive = []
            for key in fh.keys():
                which = negative if "neg" in key else positive
                val = fh[key][:]
                if val.dtype == numpy.float and len(val) == 1:
                    val = val[0]
                if val.dtype.char == "S" and val[0].decode() == "None":
                    val = None
                which.append(val)
        return list(zip(negative, positive))

    # No error files
    cmc_scores = _load_cmc_scores("test0-open-set.hdf5")
    assert (
        abs(detection_identification_rate(cmc_scores, threshold=0.5) - 1.0)
        < 1e-8
    )
    assert abs(false_alarm_rate(cmc_scores, threshold=0.5)) < 1e-8
    assert abs(recognition_rate(cmc_scores) - 7.0 / 9.0) < 1e-8
    assert abs(recognition_rate(cmc_scores, threshold=0.5) - 1.0) < 1e-8

    # One error
    cmc_scores = _load_cmc_scores("test1-open-set.hdf5")
    assert (
        abs(
            detection_identification_rate(cmc_scores, threshold=0.5) - 6.0 / 7.0
        )
        < 1e-8
    )
    assert abs(false_alarm_rate(cmc_scores, threshold=0.5)) < 1e-8
    assert abs(recognition_rate(cmc_scores) - 6.0 / 9.0) < 1e-8
    assert abs(recognition_rate(cmc_scores, threshold=0.5) - 6.0 / 7.0) < 1e-8

    # Two errors
    cmc_scores = _load_cmc_scores("test2-open-set.hdf5")
    assert (
        abs(
            detection_identification_rate(cmc_scores, threshold=0.5) - 6.0 / 7.0
        )
        < 1e-8
    )
    assert abs(false_alarm_rate(cmc_scores, threshold=0.5) - 0.5) < 1e-8
    assert abs(recognition_rate(cmc_scores) - 6.0 / 9.0) < 1e-8
    assert abs(recognition_rate(cmc_scores, threshold=0.5) - 6.0 / 8.0) < 1e-8


def test_mindcf():
    # Tests outlier scores in negative set
    cost = 0.99
    negatives = [-3, -2, -1, -0.5, 4]
    positives = [0.5, 3]
    th = min_weighted_error_rate_threshold(negatives, positives, cost, True)
    far, frr = farfrr(negatives, positives, th)
    mindcf = (cost * far + (1 - cost) * frr) * 100
    assert mindcf < 1.0 + 1e-8


def test_roc_auc_score():
    positives = _load("nonsep-positives.hdf5")
    negatives = _load("nonsep-negatives.hdf5")
    auc = roc_auc_score(negatives, positives)

    # commented out sklearn computation to avoid adding an extra test dependency
    # from sklearn.metrics import roc_auc_score as oracle_auc
    # y_true = numpy.concatenate([numpy.ones_like(positives), numpy.zeros_like(negatives)], axis=0)
    # y_score = numpy.concatenate([positives, negatives], axis=0)
    # oracle = oracle_auc(y_true, y_score)
    oracle = 0.9326

    numpy.testing.assert_allclose(
        auc, oracle
    ), f"Expected {oracle} but got {auc} instead."

    # test the function on log scale as well
    auc = roc_auc_score(negatives, positives, log_scale=True)
    oracle = 1.4183699583300993
    numpy.testing.assert_allclose(
        auc, oracle
    ), f"Expected {oracle} but got {auc} instead."
