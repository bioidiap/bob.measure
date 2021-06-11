#!/usr/bin/env python
# coding=utf-8

"""Functions for brute-force minimization leading to EER or WER thresholds.

Most of these were imported from older C++ implementations.
"""

import sys
import numpy


def _minimizing_threshold(negatives, positives, criterium):
    """Calculates the best threshold taking a predicate as input condition

    This method can calculate a threshold based on a set of scores (positives
    and negatives) given a certain minimization criterium, input as a
    functional predicate. For a discussion on ``positive`` and ``negative`` see
    :py:func:`farfrr`.  Here, it is expected that the positives and the
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

    criterium : :py:obj:`func`
        A predicate in the format ``predicate(fa_ratio, fr_ratio) -> float``


    Returns
    =======

    threshold : float
        The optimal threshold given the predicate and the scores

    """

    if not len(negatives) or not len(positives):
        raise RuntimeError(
            "Cannot compute threshold when no positives or "
            "no negatives are provided"
        )

    # N.B.: Unoptimized version ported from C++

    # iterates over all possible far and frr points and compute the predicate
    # for each possible threshold...
    min_predicate = 1e8
    min_threshold = 1e8
    current_predicate = 1e8
    # we start with the extreme values for far and frr
    far = 1.0
    frr = 0.0

    # the decrease/increase for far/frr when moving one negative/positive
    max_neg = len(negatives)
    far_decrease = 1.0 / max_neg
    max_pos = len(positives)
    frr_increase = 1.0 / max_pos

    # we start with the threshold based on the minimum score

    # iterates until one of these goes bananas
    pos_it = 0
    neg_it = 0
    current_threshold = min(negatives[neg_it], positives[pos_it])

    # continues until one of the two iterators reaches the end of the list
    while pos_it < max_pos and neg_it < max_neg:

        # compute predicate
        current_predicate = criterium(far, frr)

        if current_predicate <= min_predicate:
            min_predicate = current_predicate
            min_threshold = current_threshold

        if positives[pos_it] >= negatives[neg_it]:
            # compute current threshold
            current_threshold = negatives[neg_it]
            neg_it += 1
            far -= far_decrease

        else:  # pos_val <= neg_val
            # compute current threshold
            current_threshold = positives[pos_it]
            pos_it += 1
            frr += frr_increase

        # skip until next "different" value, which case we "gain" 1 unit on
        # the "FAR" value, since we will be accepting that negative as a
        # true negative, and not as a false positive anymore.  we continue
        # to do so for as long as the current threshold matches the current
        # iterator.
        while neg_it < max_neg and current_threshold == negatives[neg_it]:
            neg_it += 1
            far -= far_decrease

        # skip until next "different" value, which case we "loose" 1 unit
        # on the "FRR" value, since we will be accepting that positive as a
        # false negative, and not as a true positive anymore.  we continue
        # to do so for as long as the current threshold matches the current
        # iterator.
        while pos_it < max_pos and current_threshold == positives[pos_it]:
            pos_it += 1
            frr += frr_increase

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
    current_predicate = criterium(far, frr)
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
        current_predicate = criterium(0.0, 1.0)
        if current_predicate < min_predicate:
            min_predicate = current_predicate
            min_threshold = last_threshold

    # return the best threshold found
    return min_threshold


def abs_difference(a, b):
    return abs(a-b)


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

        The threshold (i.e., as used in :py:func:`farfrr`) where FPR and FNR
        are as close as possible

    """

    # if not pre-sorted, copies and sorts
    neg = negatives if is_sorted else numpy.sort(negatives)
    pos = positives if is_sorted else numpy.sort(positives)

    return _minimizing_threshold(neg, pos, abs_difference)


def far_threshold(negatives, positives, far_value=0.001, is_sorted=False):
    """Threshold such that the real FPR is **at most** the requested ``far_value`` if possible


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

    far_value : :py:class:`float`, Optional

        The FPR value, for which the threshold should be computed

    is_sorted : :py:class:`bool`, Optional

        Set this to ``True`` if the ``negatives`` are already sorted in
        ascending order.  If ``False``, scores will be sorted internally, which
        will require more memory.


    Returns
    =======

    threshold : float

        The threshold such that the real FPR is at most ``far_value``

    """

    # N.B.: Unoptimized version ported from C++

    if far_value < 0.0 or far_value > 1.0:
        raise RuntimeError(
            f"`far_value' cannot be {far_value} - "
            f"the value must be in the interval [0.,1.]"
        )

    if len(negatives) < 2:
        raise RuntimeError("the number of negative scores must be at least 2")

    # if not pre-sorted, copies and sorts
    scores = negatives if is_sorted else numpy.sort(negatives)

    # handles special case of far == 1 without any iterating
    if far_value >= (1 - sys.float_info.epsilon):
        return numpy.nextafter(scores[0], scores[0] - 1)

    # Reverse negatives so the end is the start. This way the code below will
    # be very similar to the implementation in the frr_threshold function. The
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
    future_far = 0.0

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
        future_far = (current_position + 1) / total_count
        if future_far > far_value:
            break
        valid_threshold = current_threshold
        current_position += 1

    return valid_threshold


def frr_threshold(negatives, positives, frr_value=0.001, is_sorted=False):
    """Computes the threshold such that the real FNR is **at most** the requested ``frr_value`` if possible


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

    frr_value : :py:class:`float`, Optional

        The FNR value, for which the threshold should be computed.

    is_sorted : :py:class:`bool`, Optional

        Set this to ``True`` if the ``positives`` are already sorted in
        ascending order. If ``False``, scores will be sorted internally, which
        will require more memory.


    Returns
    =======

    threshold : float

        The threshold such that the real FRR is at most ``frr_value``.

    """

    # N.B.: Unoptimized version ported from C++

    if frr_value < 0.0 or frr_value > 1.0:
        raise RuntimeError(
            f"`frr_value' cannot be {frr_value} - "
            f"the value must be in the interval [0.,1.]"
        )

    if len(positives) < 2:
        raise RuntimeError("the number of positive scores must be at least 2")

    # if not pre-sorted, copies and sorts
    scores = positives if is_sorted else numpy.sort(positives)

    # handles special case of far == 1 without any iterating
    if frr_value >= (1 - sys.float_info.epsilon):
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
    future_far = 0.0

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
        future_frr = current_position / total_count
        if future_frr > frr_value:
            break
        valid_threshold = current_threshold
        current_position += 1

    return valid_threshold


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
    return min_weighted_error_rate_threshold(
        negatives, positives, 0.5, is_sorted
    )


class _WeighedError:
    """A functor predicate for weighted error calculation"""

    def __init__(self, weight):
        self.weight = weight
        if weight > 1.0:
            self.weight = 1.0
        if weight < 0.0:
            self.weight = 0.0

    def __call__(self, far, frr):
        return (self.weight * far) + ((1.0 - self.weight) * frr)


def min_weighted_error_rate_threshold(
    negatives, positives, cost, is_sorted=False
):
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

    return _minimizing_threshold(neg, pos, _WeighedError(cost))
