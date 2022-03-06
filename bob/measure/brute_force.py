#!/usr/bin/env python
# coding=utf-8

"""Various functions for performance assessment

Most of these were imported from older C++ implementations.
"""

import numpy
import numpy.linalg
from numba import jit, objmode
import logging
from functools import wraps

logger = logging.getLogger(__name__)


def _lists_to_arrays(*args, **kwargs):
    ret, retkw = list(), dict()
    for v in args:
        ret.append(numpy.asarray(v) if isinstance(v, list) else v)
    for k, v in kwargs.items():
        retkw[k] = numpy.asarray(v) if isinstance(v, list) else v
    return ret, retkw


def array_jit(func):
    jit_func = jit(func, nopython=True)

    @wraps(jit_func)
    def new_func(*args, **kwargs):
        args, kwargs = _lists_to_arrays(*args, **kwargs)
        return jit_func(*args, **kwargs)

    new_func.jit_func = jit_func
    return new_func


@jit("float64(float64, float64, float64)", nopython=True)
def _weighted_err(fpr, fnr, cost):
    return (cost * fpr) + ((1.0 - cost) * fnr)


@jit(nopython=True)
def _minimizing_threshold(negatives, positives, criterion, cost=0.5):
    """Calculates the best threshold taking a predicate as input condition

    This method can calculate a threshold based on a set of scores (positives
    and negatives) given a certain minimization criterium, input as a
    functional predicate. For a discussion on ``positive`` and ``negative`` see
    :py:func:`fprfnr`.  Here, it is expected that the positives and the
    negatives are sorted ascendantly.

    The predicate method gives back the current minimum given false-acceptance
    (FA) and false-rejection (FR) ratios for the input data. The API for the
    criterium is:

    predicate(fa_ratio : float, fr_ratio : float) -> float

    Please note that this method will only work with single-minimum smooth
    predicates.

    The minimization is carried out in a data-driven way.  Starting from the
    lowest score (might be a positive or a negative), it increases the
    threshold based on the distance between the current score and the following
    higher score (also keeping track of duplicate scores) and computes the
    predicate for each possible threshold.

    Finally, that threshold is returned, for which the predicate returned the
    lowest value.


    Parameters
    ==========

    negatives : numpy.ndarray (1D, float)
        Negative scores, sorted ascendantly

    positives : numpy.ndarray (1D, float)
        Positive scores, sorted ascendantly

    criterion : str
        A predicate from one of ("absolute-difference", "weighted-error")

    cost : float
        Extra cost argument to be passed to criterion

    Returns
    =======

    threshold : float
        The optimal threshold given the predicate and the scores

    """
    if criterion not in ("absolute-difference", "weighted-error"):
        raise ValueError("Uknown criterion")

    def criterium(a, b, c):
        if criterion == "absolute-difference":
            return _abs_diff(a, b, c)
        else:
            return _weighted_err(a, b, c)

    if not len(negatives) or not len(positives):
        raise RuntimeError(
            "Cannot compute threshold when no positives or " "no negatives are provided"
        )

    # iterates over all possible fpr and fnr points and compute the predicate
    # for each possible threshold...
    min_predicate = 1e8
    min_threshold = 1e8
    current_predicate = 1e8
    # we start with the extreme values for fpr and fnr
    fpr = 1.0
    fnr = 0.0

    # the decrease/increase for fpr/fnr when moving one negative/positive
    max_neg = len(negatives)
    fpr_decrease = 1.0 / max_neg
    max_pos = len(positives)
    fnr_increase = 1.0 / max_pos

    # we start with the threshold based on the minimum score

    # iterates until one of these goes bananas
    pos_it = 0
    neg_it = 0
    current_threshold = min(negatives[neg_it], positives[pos_it])

    # continues until one of the two iterators reaches the end of the list
    while pos_it < max_pos and neg_it < max_neg:

        # compute predicate
        current_predicate = criterium(fpr, fnr, cost)

        if current_predicate <= min_predicate:
            min_predicate = current_predicate
            min_threshold = current_threshold

        if positives[pos_it] >= negatives[neg_it]:
            # compute current threshold
            current_threshold = negatives[neg_it]
            neg_it += 1
            fpr -= fpr_decrease

        else:  # pos_val <= neg_val
            # compute current threshold
            current_threshold = positives[pos_it]
            pos_it += 1
            fnr += fnr_increase

        # skip until next "different" value, which case we "gain" 1 unit on
        # the "FAR" value, since we will be accepting that negative as a
        # true negative, and not as a false positive anymore.  we continue
        # to do so for as long as the current threshold matches the current
        # iterator.
        while neg_it < max_neg and current_threshold == negatives[neg_it]:
            neg_it += 1
            fpr -= fpr_decrease

        # skip until next "different" value, which case we "loose" 1 unit
        # on the "FRR" value, since we will be accepting that positive as a
        # false negative, and not as a true positive anymore.  we continue
        # to do so for as long as the current threshold matches the current
        # iterator.
        while pos_it < max_pos and current_threshold == positives[pos_it]:
            pos_it += 1
            fnr += fnr_increase

        # computes a new threshold based on the center between last and current
        # score, if we are **not** already at the end of the score lists
        if neg_it < max_neg or pos_it < max_pos:
            if neg_it < max_neg and pos_it < max_pos:
                current_threshold += min(negatives[neg_it], positives[pos_it])
            elif neg_it < max_neg:
                current_threshold += negatives[neg_it]
            else:
                current_threshold += positives[pos_it]
            current_threshold /= 2

    # now, we have reached the end of one list (usually the negatives) so,
    # finally compute predicate for the last time
    current_predicate = criterium(fpr, fnr, cost)
    if current_predicate < min_predicate:
        min_predicate = current_predicate
        min_threshold = current_threshold

    # now we just double check choosing the threshold higher than all scores
    # will not improve the min_predicate
    if neg_it < max_neg or pos_it < max_pos:
        last_threshold = current_threshold
        if neg_it < max_neg:
            last_threshold = numpy.nextafter(negatives[-1], negatives[-1] + 1)
        elif pos_it < max_pos:
            last_threshold = numpy.nextafter(positives[-1], positives[-1] + 1)
        current_predicate = criterium(0.0, 1.0, cost)
        if current_predicate < min_predicate:
            min_predicate = current_predicate
            min_threshold = last_threshold

    # return the best threshold found
    return min_threshold


@jit("float64(float64, float64, float64)", nopython=True)
def _abs_diff(a, b, cost):
    return abs(a - b)


def eer_threshold(negatives, positives, is_sorted=False):
    """Calculates threshold as close as possible to the equal error rate (EER)

    The EER should be the point where the FPR equals the FNR. Graphically, this
    would be equivalent to the intersection between the ROC (or DET) curves and
    the identity.

    .. note::

       The scores will be sorted internally, requiring the scores to be copied.
       To avoid this copy, you can sort both sets of scores externally in
       ascendant order, and set the ``is_sorted`` parameter to ``True``.


    Parameters
    ==========

    negatives : numpy.ndarray (1D, float)

        The set of negative scores to compute the threshold

    positives : numpy.ndarray (1D, float)

        The set of positive scores to compute the threshold

    is_sorted : :py:class:`bool`, Optional

        Set this to ``True`` if the ``negatives`` are already sorted in
        ascending order.  If ``False``, scores will be sorted internally, which
        will require more memory.


    Returns
    =======

    threshold : float

        The threshold (i.e., as used in :py:func:`fprfnr`) where FPR and FNR
        are as close as possible

    """

    # if not pre-sorted, copies and sorts
    neg = negatives if is_sorted else numpy.sort(negatives)
    pos = positives if is_sorted else numpy.sort(positives)

    return _minimizing_threshold(neg, pos, "absolute-difference")


@array_jit
def fpr_threshold(negatives, positives, fpr_value=0.001, is_sorted=False):
    """Threshold such that the real FPR is **at most** the requested ``fpr_value`` if possible


    .. note::

       The scores will be sorted internally, requiring the scores to be copied.
       To avoid this copy, you can sort the ``negatives`` scores externally in
       ascendant order, and set the ``is_sorted`` parameter to ``True``.


    Parameters
    ==========

    negatives : numpy.ndarray (1D, float)

        The set of negative scores to compute the FPR threshold

    positives : numpy.ndarray (1D, float)

        Ignored, but needs to be specified -- may be given as ``[]``

    fpr_value : :py:class:`float`, Optional

        The FPR value, for which the threshold should be computed

    is_sorted : :py:class:`bool`, Optional

        Set this to ``True`` if the ``negatives`` are already sorted in
        ascending order.  If ``False``, scores will be sorted internally, which
        will require more memory.


    Returns
    =======

    threshold : float

        The threshold such that the real FPR is at most ``fpr_value``

    """

    # N.B.: Unoptimized version ported from C++

    if fpr_value < 0.0 or fpr_value > 1.0:
        raise RuntimeError("`fpr_value' must be in the interval [0.,1.]")

    if len(negatives) < 2:
        raise RuntimeError("the number of negative scores must be at least 2")

    epsilon = numpy.finfo(numpy.float64).eps
    # if not pre-sorted, copies and sorts
    scores = negatives if is_sorted else numpy.sort(negatives)

    # handles special case of fpr == 1 without any iterating
    if fpr_value >= (1 - epsilon):
        return numpy.nextafter(scores[0], scores[0] - 1)

    # Reverse negatives so the end is the start. This way the code below will
    # be very similar to the implementation in the fnr_threshold function. The
    # implementations are not exactly the same though.
    scores = numpy.flip(scores)

    # Move towards the end of array changing the threshold until we cross the
    # desired FAR value. Starting with a threshold that corresponds to FAR ==
    # 0.
    total_count = len(scores)
    current_position = 0

    # since the comparison is `if score >= threshold then accept as genuine`,
    # we can choose the largest score value + eps as the threshold so that we
    # can get for 0% FAR.
    valid_threshold = numpy.nextafter(
        scores[current_position], scores[current_position] + 1
    )
    current_threshold = 0.0

    while current_position < total_count:

        current_threshold = scores[current_position]
        # keep iterating if values are repeated
        while (
            current_position < (total_count - 1)
            and scores[current_position + 1] == current_threshold
        ):
            current_position += 1
        # All the scores up to the current position and including the current
        # position will be accepted falsely.
        future_fpr = (current_position + 1) / total_count
        if future_fpr > fpr_value:
            break
        valid_threshold = current_threshold
        current_position += 1

    return valid_threshold


_jit_fpr_threshold = fpr_threshold.jit_func


far_threshold = fpr_threshold


@array_jit
def fnr_threshold(negatives, positives, fnr_value=0.001, is_sorted=False):
    """Computes the threshold such that the real FNR is **at most** the requested ``fnr_value`` if possible


    .. note::

       The scores will be sorted internally, requiring the scores to be copied.
       To avoid this copy, you can sort the ``positives`` scores externally in
       ascendant order, and set the ``is_sorted`` parameter to ``True``.


    Parameters
    ==========

    negatives : numpy.ndarray (1D, float)

        Ignored, but needs to be specified -- may be given as ``[]``.

    positives : numpy.ndarray (1D, float)

        The set of positive scores to compute the FNR threshold.

    fnr_value : :py:class:`float`, Optional

        The FNR value, for which the threshold should be computed.

    is_sorted : :py:class:`bool`, Optional

        Set this to ``True`` if the ``positives`` are already sorted in
        ascending order. If ``False``, scores will be sorted internally, which
        will require more memory.


    Returns
    =======

    threshold : float

        The threshold such that the real FRR is at most ``fnr_value``.

    """

    # N.B.: Unoptimized version ported from C++

    if fnr_value < 0.0 or fnr_value > 1.0:
        raise RuntimeError("`fnr_value' value must be in the interval [0.,1.]")

    if len(positives) < 2:
        raise RuntimeError("the number of positive scores must be at least 2")

    epsilon = numpy.finfo(numpy.float64).eps
    # if not pre-sorted, copies and sorts
    scores = positives if is_sorted else numpy.sort(positives)

    # handles special case of fpr == 1 without any iterating
    if fnr_value >= (1 - epsilon):
        return numpy.nextafter(scores[-1], scores[-1] + 1)

    # Move towards the end of array changing the threshold until we cross the
    # desired FRR value. Starting with a threshold that corresponds to FRR ==
    # 0.
    total_count = len(scores)
    current_position = 0

    # since the comparison is `if score >= threshold then accept as genuine`,
    # we can choose the largest score value + eps as the threshold so that we
    # can get for 0% FAR.
    valid_threshold = numpy.nextafter(
        scores[current_position], scores[current_position] + 1
    )
    current_threshold = 0.0

    while current_position < total_count:

        current_threshold = scores[current_position]
        # keep iterating if values are repeated
        while (
            current_position < (total_count - 1)
            and scores[current_position + 1] == current_threshold
        ):
            current_position += 1
        # All the scores up to the current position and including the current
        # position will be accepted falsely.
        future_fnr = current_position / total_count
        if future_fnr > fnr_value:
            break
        valid_threshold = current_threshold
        current_position += 1

    return valid_threshold


_jit_fnr_threshold = fnr_threshold.jit_func


frr_threshold = fnr_threshold


def min_hter_threshold(negatives, positives, is_sorted=False):
    """Calculates the :py:func:`min_weighted_error_rate_threshold` with ``cost=0.5``

    Parameters
    ==========

    negatives, positives : numpy.ndarray (1D, float)

        The set of negative and positive scores to compute the threshold

    is_sorted : :py:class:`bool`, Optional

        Set this to ``True`` if the ``positives`` are already sorted in
        ascending order. If ``False``, scores will be sorted internally, which
        will require more memory.


    Returns
    =======

    threshold : float

        The threshold for which the weighted error rate is minimal

    """
    return min_weighted_error_rate_threshold(negatives, positives, 0.5, is_sorted)


@array_jit
def min_weighted_error_rate_threshold(negatives, positives, cost, is_sorted=False):
    """Calculates the threshold that minimizes the error rate

    The ``cost`` parameter determines the relative importance between
    false-accepts and false-rejections. This number should be between 0 and 1
    and will be clipped to those extremes. The value to minimize becomes:
    :math:`ER_{cost} = cost * FPR + (1-cost) * FNR`. The higher the cost, the
    higher the importance given to **not** making mistakes classifying
    negatives/noise/impostors.

    .. note::

       The scores will be sorted internally, requiring the scores to be copied.
       To avoid this copy, you can sort both sets of scores externally in
       ascendant order, and set the ``is_sorted`` parameter to ``True``.


    Parameters
    ==========

    negatives, positives : numpy.ndarray (1D, float)

        The set of negative and positive scores to compute the threshold

    cost : float

        The relative cost over FPR with respect to FNR in the threshold
        calculation

    is_sorted : :py:class:`bool`, Optional

        Set this to ``True`` if the ``positives`` are already sorted in
        ascending order. If ``False``, scores will be sorted internally, which
        will require more memory.


    Returns
    =======

    threshold : float

        The threshold for which the weighted error rate is minimal

    """

    # if not pre-sorted, copies and sorts
    neg = negatives if is_sorted else numpy.sort(negatives)
    pos = positives if is_sorted else numpy.sort(positives)
    if cost > 1.0:
        cost = 1.0
    elif cost < 0.0:
        cost = 0.0

    return _minimizing_threshold(neg, pos, "weighted-error", cost)


_jit_min_weighted_error_rate_threshold = min_weighted_error_rate_threshold.jit_func
