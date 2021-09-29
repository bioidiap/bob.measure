#!/usr/bin/env python
# coding=utf-8

"""Curve plotting support

Code in this module expresses classic performance curves for system performance
evaluation as sets of coordinates.  Use the module :py:mod:`plot` to make
graphical representations.

.. include:: ../links.rst
"""

import sys
import numpy
import numpy.linalg

from .brute_force import far_threshold, frr_threshold


def _log_values(points, min_power):
    """Computes log-scaled values between :math:`10^\text{min_power}` and 1

    This function returns

    Parameters
    ==========

    points : int
        Number of points to consider

    min_power : int
        Negative integer with the minimum power


    Returns
    =======

    logscale : numpy.ndarray (float, 1D)

        A set of numbers forming a logarithm-based scale between
        :math:`10^\text{min_power}` and 1.

    """

    return 10 ** (numpy.arange(1 - points, 1) / int(points / (-min_power)))


def _meaningful_thresholds(negatives, positives, n_points, min_fpr, is_sorted):
    """Returns non-repeatitive thresholds to generate ROC curves

    This function creates a list of FAR (and FRR) values that we are
    interesting to see on the curve.  Computes thresholds for those points.
    Sorts the thresholds so we get sorted numbers to plot on the curve and
    returns the thresholds.  Some points will be duplicate but in terms of
    resolution and accuracy this is better than just changing the threshold
    from ``min()`` of scores to ``max()`` of scores with equal spaces.


    Parameters
    ==========

    negatives, positives : numpy.ndarray (1D, float)

        The negative and positive scores, for which the meaningful threshold
        will be calculated.

    n_points : int

        The number of points, in which the ROC curve are calculated, which are
        distributed uniformly in the range ``[min(negatives, positives),
        max(negatives, positives)]``

    min_fpr : int

        Minimum FPR in terms of :math:`10^(\text{min_fpr}`. This value is also
        used for ``min_fnr``. Values should be negative.


    is_sorted : bool

        Set this to ``True`` if both sets of scores are already sorted in
        ascending order.  If ``False``, scores will be sorted internally, which
        will require more memory.


    Returns
    =======

    thresholds : numpy.ndarray (1D, float)

        The "meaningful" thresholds that would cause changes in the ROC.

    """

    half_points = n_points // 2

    # if not pre-sorted, copies and sorts
    neg = negatives if is_sorted else numpy.sort(negatives)
    pos = positives if is_sorted else numpy.sort(positives)

    fpr_list = _log_values(n_points - half_points, min_fpr)
    fnr_list = _log_values(half_points, min_fpr)

    t = numpy.zeros(
        [
            n_points,
        ],
        dtype=float,
    )
    t[:half_points] = [frr_threshold(neg, pos, k, True) for k in fnr_list]
    t[half_points:] = [far_threshold(neg, pos, k, True) for k in fpr_list]

    t.sort()

    return t


def roc(negatives, positives, n_points, min_fpr=-8):
    """Calculates points of an Receiver Operating Characteristic (ROC)

    Calculates the ROC curve (false-positive rate against false-negative rate)
    given a set of negative and positive scores and a desired number of points.


    Parameters
    ==========

    negatives, positives : numpy.ndarray (1D, float)

        The negative and positive scores, for which the ROC curve should be
        calculated.

    n_points : int

        The number of points, in which the ROC curve are calculated, which are
        distributed uniformly in the range ``[min(negatives, positives),
        max(negatives, positives)]``

    min_fpr : int

        Minimum FAR in terms of :math:`10^(\text{min_fpr}`. This value is also
        used for ``min_fnr``. Values should be negative.


    Returns
    =======

    curve : numpy.ndarray (2D, float)

        A two-dimensional array of doubles that express the FPR and FNR
        coordinates in this order

    """
    from .binary import fprfnr

    t = _meaningful_thresholds(negatives, positives, n_points, min_fpr, False)
    return numpy.array([fprfnr(negatives, positives, k) for k in t]).T


def det(negatives, positives, n_points, min_fpr=-8):
    """Calculates points of an Detection Error-Tradeoff (DET) curve

    Calculates the DET curve given a set of negative and positive scores and a
    desired number of points. Returns a two-dimensional array of doubles that
    express on its rows:

    You can plot the results using your preferred tool to first create a plot
    using rows 0 and 1 from the returned value and then replace the X/Y axis
    annotation using a pre-determined set of tickmarks as recommended by NIST.
    The derivative scales are computed with the :py:func:`ppndf` function.


    Parameters
    ==========

    negatives, positives : numpy.ndarray (1D, float)

        The list of negative and positive scores to compute the DET for

    n_points : int

        The number of points on the DET curve, for which the DET should be
        evaluated

    min_fpr : :class:`int`, Optional

        Minimum FAR in terms of :math:`10^\text{min_fpr}`. This value is also
        used for ``min_fnr``. Values should be negative.


    Returns
    =======

    curve : numpy.ndarray (2D, float)

        The DET curve, with the FPR in the first and the FNR in the second
        row:

        0. X axis values in the normal deviate scale for the false-positives
        1. Y axis values in the normal deviate scale for the false-negatives

    """
    return ppndf(roc(negatives, positives, n_points, min_fpr))


def precision_recall(negatives, positives, n_points):
    """Calculates the precision-recall curve given a set of positive and negative scores and a number of desired points

    The points in which the curve is calculated are distributed uniformly in
    the range ``[min(negatives, positives), max(negatives, positives)]``


    Parameters
    ==========

    negatives, positives : numpy.ndarray (1D, float)

        The set of negative and positive scores to compute the measurements

    n_points : int

        The number of thresholds for which precision and recall should be
        evaluated


    Returns
    =======

    curve : numpy.ndarray (2D, float)

        2D array of floats that express the X (precision) and Y (recall)
        coordinates.

    """
    from .binary import precision_recall

    # evaluates all interesting thresholds worth plotting
    return numpy.array(
        [
            precision_recall(negatives, positives, k)
            for k in _meaningful_thresholds(
                negatives, positives, n_points, -8, False
            )
        ]
    ).T


def roc_for_far(negatives, positives, fpr_list, is_sorted=False):
    """Calculates the ROC curve for a given set of positive and negative scores and the FPR values, for which the FNR should be computed

    .. note::

       The scores will be sorted internally, requiring the scores to be copied.
       To avoid this copy, you can sort both sets of scores externally in
       ascendant order, and set the ``is_sorted`` parameter to ``True``.


    Parameters
    ==========

    negatives, positives : numpy.ndarray (1D, float)

        The set of negative and positive scores to compute the curve

    fpr_list : numpy.ndarray (1D, float)

        A list of FPR values, for which the FNR values should be computed

    is_sorted : :py:class:`bool`, Optional

        Set this to ``True`` if both sets of scores are already sorted in
        ascending order.  If ``False``, scores will be sorted internally, which
        will require more memory.


    Returns
    =======

    curve : numpy.ndarray (2D, float)

        The ROC curve, which holds a copy of the given FPR values in row 0, and
        the corresponding FNR values in row 1.

    """
    from .binary import fprfnr

    n_points = len(fpr_list)

    if len(negatives) == 0:
        raise RuntimeError("The given set of negatives is empty.")

    if len(positives) == 0:
        raise RuntimeError("The given set of positives is empty.")

    # if not pre-sorted, copies and sorts
    neg = negatives if is_sorted else numpy.sort(negatives)
    pos = positives if is_sorted else numpy.sort(positives)

    # Get the threshold for the requested far values and calculate fpr and fnr
    # values based on the threshold.
    return numpy.array(
        [fprfnr(neg, pos, far_threshold(neg, pos, k, True)) for k in fpr_list]
    ).T


def ppndf(p):
    """Returns the Deviate Scale equivalent of a false rejection/acceptance ratio

    The algorithm that calculates the deviate scale is based on function
    ``ppndf()`` from the NIST package DETware version 2.1, freely available on
    the internet. Please consult it for more details. By 20.04.2011, you could
    find such package `here <http://www.itl.nist.gov/iad/mig/tools/>`_.

    The input to this function is a cumulative probability.  The output from
    this function is the Normal deviate that corresponds to that probability.
    For example:

    ======= ========
     INPUT   OUTPUT
    ======= ========
     0.001   -3.090
     0.01    -2.326
     0.1     -1.282
     0.5      0.0
     0.9      1.282
     0.99     2.326
     0.999    3.090
    ======= ========


    Parameters
    ==========

    p : numpy.ndarray (2D, float)

        The value (usually FPR or FNR) for which the PPNDF should be calculated


    Returns
    =======

    ppndf : numpy.ndarray (2D, float)

        The derivative scale of the given value

    """

    # threshold
    p = numpy.array(p, dtype=float)
    p[p >= 1.0] = 1.0 - sys.float_info.epsilon
    p[p <= 0.0] = sys.float_info.epsilon

    q = p - 0.5
    abs_q = numpy.abs(q)

    retval = numpy.zeros_like(p)

    # first part q<=0.42
    q1 = q[abs_q <= 0.42]
    r = numpy.square(q1)
    opt1 = (
        q1
        * (
            ((-25.4410604963 * r + 41.3911977353) * r + -18.6150006252) * r
            + 2.5066282388
        )
        / (
            (
                ((3.1308290983 * r + -21.0622410182) * r + 23.0833674374) * r
                + -8.4735109309
            )
            * r
            + 1.0
        )
    )
    retval[abs_q <= 0.42] = opt1

    # second part q>0.42
    # r = sqrt (log (0.5 - abs(q)));
    q2 = q[abs_q > 0.42]
    r = p[abs_q > 0.42]
    r[q2 > 0] = 1 - r[q2 > 0]
    if (r <= 0).any():
        raise RuntimeError("measure::ppndf(): r <= 0.0!")

    r = numpy.sqrt(-1 * numpy.log(r))
    opt2 = (
        ((2.3212127685 * r + 4.8501412713) * r + -2.2979647913) * r
        + -2.7871893113
    ) / ((1.6370678189 * r + 3.5438892476) * r + 1.0)
    opt2[q2 < 0] *= -1
    retval[abs_q > 0.42] = opt2

    return retval


def epc(
    dev_negatives,
    dev_positives,
    test_negatives,
    test_positives,
    n_points,
    is_sorted=False,
    thresholds=False,
):
    """Calculates points of an Expected Performance Curve (EPC)

    Calculates the EPC curve given a set of positive and negative scores and a
    desired number of points. Returns a two-dimensional
    :py:class:`numpy.ndarray` of type float with the shape of ``(2, points)``
    or ``(3, points)`` depending on the ``thresholds`` argument.  The rows
    correspond to the X (cost), Y (weighted error rate on the test set given
    the min. threshold on the development set), and the thresholds which were
    used to calculate the error (if the ``thresholds`` argument was set to
    ``True``), respectively. Please note that, in order to calculate the EPC
    curve, one needs two sets of data comprising a development set and a test
    set. The minimum weighted error is calculated on the development set and
    then applied to the test set to evaluate the weighted error rate at that
    position.

    The EPC curve plots the HTER on the test set for various values of 'cost'.
    For each value of 'cost', a threshold is found that provides the minimum
    weighted error (see :py:func:`min_weighted_error_rate_threshold`) on the
    development set. Each threshold is consecutively applied to the test set
    and the resulting weighted error values are plotted in the EPC.

    The cost points in which the EPC curve are calculated are distributed
    uniformly in the range :math:`[0.0, 1.0]`.

    .. note::

       It is more memory efficient, when sorted arrays of scores are provided
       and the ``is_sorted`` parameter is set to ``True``.


    Parameters
    ==========

    dev_negatives : numpy.ndarray (1D, float)

        The scores for non-target objects, or generated by comparing objects of
        different classes, on the development (or validation) set

    dev_positives : numpy.ndarray (1D, float)

        The scores for target objects, or generated by comparing objects of
        the same classe, on the development (or validation) set

    test_negatives : numpy.ndarray (1D, float)

        The scores for non-target objects, or generated by comparing objects of
        different classes, on the test set

    test_positives :  numpy.ndarray (1D, float)

        The scores for target objects, or generated by comparing objects of
        the same classe, on the test set

    n_points : int

        The number of weights for which the EPC curve should be computed

    is_sorted : :py:class:`bool`, Optional

        Set this to ``True`` if the ``negatives`` are already sorted in
        ascending order.  If ``False``, scores will be sorted internally, which
        will require more memory.

    thresholds : :py:class:`bool`, Optional

        If ``True`` the function returns an array with the shape of ``(3,
        points)`` where the third row contains the thresholds that were
        calculated on the development set.


    Returns
    =======

    curve : numpy.ndarray (2D or 3D, float)

        The EPC curve, with the first row containing the weights and the second
        row containing the weighted errors on the test set.  If ``thresholds``
        is ``True``, there is also a third row which contains the thresholds
        that were calculated on the development set.

    """
    from .brute_force import min_weighted_error_rate_threshold
    from .binary import fprfnr

    # if not pre-sorted, copies and sorts
    dev_neg = dev_negatives if is_sorted else numpy.sort(dev_negatives)
    dev_pos = dev_positives if is_sorted else numpy.sort(dev_positives)
    step = 1.0 / (n_points - 1.0)
    alpha = numpy.arange(0, 1 + step, step, dtype=float)
    thres = [
        min_weighted_error_rate_threshold(dev_neg, dev_pos, k, True)
        for k in alpha
    ]
    mwer = [
        numpy.mean(fprfnr(test_negatives, test_positives, k)) for k in thres
    ]

    if thresholds:
        return numpy.vstack([alpha, mwer, thres])
    return numpy.vstack([alpha, mwer])


def eer_threshold_from_roc(curve):
    """Calculates the equal-error-rate threshold from a (pre-computed) ROC curve

    This method tries to estimate the equal-error rate (EER) from a curve
    (instead of using brute-force minimization proposed in other modules of
    this package).

    The ROC curve is composed of rates measuring error or success.  This method
    only compares these rates until it finds the location where rates are
    equal.  If such a point does not exist considering the values on the curve,
    then it is linearly interpolated from the closest bracket.

    .. warning::

       This is a brute-force method, which may **not** be well adapted to
       fine-grained curves.


    Parameters
    ----------

    curve : numpy.ndarray (2D, float)

        A 2D numpy array of floats representing the binary classification
        performance ROC curve, that is precomputed.  It is expected that the
        first dimension of this array encodes error or success **rates**.  The
        second dimension contains the points of the curve.

        The curve object must be sorted so that values are either represented
        in either ascending or descending order.  Either success or error rates
        should be present.  You **cannot** mix them on the same curve.  For
        example, either combine FPR and FNR, or TNR and TPR.  Do **not**
        combine FPR with TPR or this method will not work!


    Returns
    -------

    threshold : float

        The computed threshold, at which the EER can be obtained

    """

    def monotonic(x):
        dx = np.diff(x)
        return np.all(dx <= 0) or np.all(dx >= 0)

    assert curve.shape[0] == 2  # error or success rates
    N = curve.shape[1]  # number of points on the curve
    assert monotonic(curve)

    threshold = 0.0  # the threshold that will be returned
    one = numpy.ones((2,), dtype=float)
    threshold_candidate = 0.0

    for i in range(N - 1):
        XY = curve[:, i : (i + 2)].T

        # computes the width and height of this segment
        if (XY[0] - XY[1]).abs().min() < sys.float_info.epsilon:
            # segment is too small, disconsider this candidate...
            continue

        # finds line coefficients seg s.t. XY.seg = 1
        seg = numpy.linalg.solve(XY, one)
        # candidate for the EER threshold (to be compared to current value)
        threshold_candidate = 1.0 / seg.sum()
        threshold = max(threshold, threshold_candidate)

    return threshold


def eer_threshold_from_rocch(negatives, positives):
    """Equal-error-rate (EER) given the input data, on the ROC Convex Hull

    It replicates the EER calculation from the Bosaris toolkit
    (https://sites.google.com/site/bosaristoolkit/).


    Parameters
    ==========

    negatives : numpy.ndarray (1D, float)

        The scores for non-target objects, or generated by comparing objects of
        different classes

    positives : numpy.ndarray (1D, float)

        The scores for target objects, or generated by comparing objects of
        the same classe


    Returns
    =======

    threshold : float

        The threshold for the equal error rate

    """
    return eer_threshold_from_roc(rocch(negatives, positives))


def rocch(negatives, positives):
    """Calculates the ROC Convex Hull (ROCCH) curve given a set of positive and negative scores


    Parameters
    ==========

    negatives, positives : numpy.ndarray (1D, float)

        The set of negative and positive scores to compute the curve


    Returns
    =======

    curve : numpy.ndarray (2D, float)

        The ROC curve, with the first row containing the FPR, and the second
        row containing the FNR.

    """

    from bob.math import pavxWidth

    # Number of positive and negative scores
    Nt = len(positives)
    Nn = len(negatives)
    N = Nt + Nn

    # Creates a big array with all scores
    scores = numpy.concatenate((positives, negatives))

    # It is important here that scores that are the same (i.e. already in
    # order) should NOT be swapped.  "stable" has this property.
    perturb = numpy.argsort(scores, kind="stable")

    # Apply permutation
    Pideal = numpy.zeros((N,), dtype=float)
    Pideal[perturb < Nt] = 1.0

    # Applies the PAVA algorithm
    Popt = numpy.ndarray((N,), dtype=float)
    width = pavxWidth(Pideal, Popt)

    # Allocates output
    nbins = len(width)
    retval = numpy.zeros((2, nbins + 1), dtype=float)  # FAR, FRR

    # Fills in output
    left = 0
    fa = Nn
    miss = 0

    for i in range(nbins):

        retval[0, i] = fa / Nn  # pfa
        retval[1, i] = miss / Nt  # pmiss
        left += int(width[i])

        if left >= 1:
            miss = Pideal[:left].sum()
        else:
            miss = 0.0

        if Pideal.shape[0] - 1 >= left:
            fa = N - left - Pideal[left:].sum()
        else:
            fa = 0

    retval[0, nbins] = fa / Nn  # pfa
    retval[1, nbins] = miss / Nt  # pmiss

    return retval


def roc_auc_score(
    negatives, positives, n_points=2000, min_fpr=-8, log_scale=False
):
    """Area Under the ROC Curve.

    Computes the area under the ROC curve. This is useful when you want to
    report one number that represents an ROC curve. This implementation uses
    the trapezoidal rule for the integration of the ROC curve. For more
    information, see:
    https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve


    Parameters
    ----------

    negatives, positives : numpy.ndarray
        The negative and positive scores, for which the meaningful threshold
        will be calculated.

    n_points : int
        The number of points, in which the ROC curve are calculated, which are
        distributed uniformly in the range ``[min(negatives, positives),
        max(negatives, positives)]``. Higher numbers leads to more accurate
        ROC.

    min_fpr : float, optional
        Min FPR and FNR values to consider when calculating ROC.

    log_scale : bool, optional
        If True, converts the x axis (FPR) to log10 scale before calculating
        AUC. This is useful in cases where len(negatives) >> len(positives)

    Returns
    -------

    auc : float
        The area under the ROC curve. If ``log_scale`` is False, the value
        should be between 0 and 1.

    """

    fpr, fnr = roc(negatives, positives, n_points, min_fpr=min_fpr)
    tpr = 1 - fnr

    if log_scale:
        fpr_pos = fpr > 0
        fpr, tpr = fpr[fpr_pos], tpr[fpr_pos]
        fpr = numpy.log10(fpr)

    area = -1 * numpy.trapz(tpr, fpr)
    return area


def estimated_ci_coverage(f, n=100, expected_coverage=0.95):
    """Returns the approximate coverage of a credible region or confidence
    interval estimator

    Reference: `This blog post <ci-evaluation_>`_.


    Parameters
    ==========

    f : object
        A callable that accepts ``k``, the number of successes (1D integer
        numpy.ndarray), ``l`` (1D integer numpy.ndarray), the number of
        failures to account for in the estimation of the interval/region, and
        ``coverage`` the coverage that is expected by this interval.  This
        function must return two float parameters only corresponding to the
        lower and upper bounds of the credible region or confidence interval
        being estimated.

    n : int
        The number of bernoulli trials to consider on the binomial
        distribution.  This represents the total number of samples you'd have
        for your experiment.

    expected_coverage : float
        A floating-point number between 0 and 1.0 indicating the coverage
        you're expecting.  A value of 0.95 will run this procedure assuming a
        95% of coverage area for the confidence interval or credible region.


    Returns
    =======

    coverage : numpy.ndarray (2D, float)
        The actual coverage curve, you can expect.  The first row corresponds
        to the values of ``p`` that were probed.  The second row, the actual
        coverage considering a simulated binomial distribution with size
        ``n``.

    """

    coverage = []
    size = 10000  # how many experiments to do at each try
    r = numpy.arange(0.001, 1.0, step=0.01)

    for p in r:
        k = numpy.random.binomial(n=n, p=p, size=size)
        covered = numpy.zeros((n,), dtype=bool)
        regions = f(k, n - k, expected_coverage)
        covered = numpy.asarray(
            (regions[0] < p) & (p < regions[1]), dtype=float
        )
        coverage.append(covered.mean())

    return numpy.vstack((r, numpy.asarray(coverage, dtype=float)))
