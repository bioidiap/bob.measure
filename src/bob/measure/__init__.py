# import Libraries of other lib packages
import numpy

from . import calibration, load, plot  # noqa: F401
from ._library import *  # noqa: F401, F403
from ._library import eer_threshold, farfrr, logger, roc


def fprfnr(negatives, positives, threshold):
    """Alias for :py:func:`bob.measure.farfrr`"""
    return farfrr(negatives, positives, threshold)


def mse(estimation, target):
    r"""Mean square error between a set of outputs and target values

    Uses the formula:

    .. math::

      MSE(\hat{\Theta}) = E[(\hat{\Theta} - \Theta)^2]

    Estimation (:math:`\hat{\Theta}`) and target (:math:`\Theta`) are supposed to
    have 2 dimensions. Different examples are organized as rows while different
    features in the estimated values or targets are organized as different
    columns.


    Parameters:

      estimation (array): an N-dimensional array that corresponds to the value
        estimated by your procedure

      target (array): an N-dimensional array that corresponds to the expected
        value


    Returns:

      float: The average of the squared error between the estimated value and the
      target

    """
    return numpy.mean((estimation - target) ** 2, 0)


def rmse(estimation, target):
    r"""Calculates the root mean square error between a set of outputs and target

    Uses the formula:

    .. math::

      RMSE(\hat{\Theta}) = \sqrt(E[(\hat{\Theta} - \Theta)^2])

    Estimation (:math:`\hat{\Theta}`) and target (:math:`\Theta`) are supposed to
    have 2 dimensions. Different examples are organized as rows while different
    features in the estimated values or targets are organized as different
    columns.


    Parameters:

      estimation (array): an N-dimensional array that corresponds to the value
        estimated by your procedure

      target (array): an N-dimensional array that corresponds to the expected
        value


    Returns:

      float: The square-root of the average of the squared error between the
      estimated value and the target

    """
    return numpy.sqrt(mse(estimation, target))


def relevance(input, machine):
    """Calculates the relevance of every input feature to the estimation process

    Uses the formula:

      Neural Triggering System Operating on High Resolution Calorimetry
      Information, Anjos et al, April 2006, Nuclear Instruments and Methods in
      Physics Research, volume 559, pages 134-138

    .. math::

      R(x_{i}) = |E[(o(x) - o(x|x_{i}=E[x_{i}]))^2]|

    In other words, the relevance of a certain input feature **i** is the change
    on the machine output value when such feature is replaced by its mean for all
    input vectors. For this to work, the `input` parameter has to be a 2D array
    with features arranged column-wise while different examples are arranged
    row-wise.


    Parameters:

      input (array): an N-dimensional array that corresponds to the value
        estimated by your model

      machine (object): A machine that can be called to "process" your input


    Returns:

      array: An 1D float array as large as the number of columns (second
      dimension) of your input array, estimating the "relevance" of each input
      column (or feature) to the score provided by the machine.

    """

    o = machine(input)
    i2 = input.copy()
    retval = numpy.ndarray((input.shape[1],), "float64")
    retval.fill(0)
    for k in range(input.shape[1]):
        i2[:, :] = input  # reset
        i2[:, k] = numpy.mean(input[:, k])
        retval[k] = (mse(machine(i2), o).sum()) ** 0.5

    return retval


def recognition_rate(cmc_scores, threshold=None, rank=1):
    """Calculates the recognition rate from the given input

    It is identical to the CMC value for the given ``rank``.

    The input has a specific format, which is a list of two-element tuples.  Each
    of the tuples contains the negative :math:`\\{S_p^-\\}` and the positive
    :math:`\\{S_p^+\\}` scores for one probe item :math:`p`, or ``None`` in case
    of open set recognition.

    If ``threshold`` is set to ``None``, the rank 1 recognition rate is defined
    as the number of test items, for which the highest positive
    :math:`\\max\\{S_p^+\\}` score is greater than or equal to all negative
    scores, divided by the number of all probe items :math:`P`:

    .. math::

      \\mathrm{RR} = \\frac{1}{P} \\sum_{p=1}^{P} \\begin{cases} 1 & \\mathrm{if } \\max\\{S_p^+\\} >= \\max\\{S_p^-\\}\\\\ 0 & \\mathrm{otherwise} \\end{cases}

    For a given rank :math:`r>1`, up to :math:`r` negative scores that are higher
    than the highest positive score are allowed to still count as correctly
    classified in the top :math:`r` rank.

    If ``threshold`` :math:`\\theta` is given, **all** scores below threshold
    will be filtered out.  Hence, if all positive scores are below threshold
    :math:`\\max\\{S_p^+\\} < \\theta`, the probe will be misclassified **at any
    rank**.

    For open set recognition, i.e., when there exist a tuple including negative
    scores without corresponding positive scores (``None``), and **all** negative
    scores are below ``threshold`` :math:`\\max\\{S_p^+\\} < \\theta`, the probe
    item is correctly rejected, **and it does not count into the denominator**
    :math:`P`.  When no ``threshold`` is provided, the open set probes will
    **always** count as misclassified, regardless of the ``rank``.

    .. warn:
       For open set tests, this rate does not correspond to a standard rate.
       Please use :py:func:`detection_identification_rate` and
       :py:func:`false_alarm_rate` instead.


    Parameters:

      cmc_scores (:py:class:`list`): A list in the format ``[(negatives,
        positives), ...]`` containing the CMC scores (i.e. :py:class:`list`:
        A list of tuples, where each tuple contains the
        ``negative`` and ``positive`` scores for one probe of the database).

        Each pair contains the ``negative`` and the ``positive`` scores for **one
        probe item**.  Each pair can contain up to one empty array (or ``None``),
        i.e., in case of open set recognition.

      threshold (:obj:`float`, optional): Decision threshold. If not ``None``, **all**
        scores will be filtered by the threshold. In an open set recognition
        problem, all open set scores (negatives with no corresponding positive)
        for which all scores are below threshold, will be counted as correctly
        rejected and **removed** from the probe list (i.e., the denominator).

      rank (:obj:`int`, optional):
        The rank for which the recognition rate should be computed, 1 by default.


    Returns:

      float: The (open set) recognition rate for the given rank, a value between
      0 and 1.

    """
    # If no scores are given, the recognition rate is exactly 0.
    if not cmc_scores:
        return 0.0

    correct = 0
    counter = 0
    for neg, pos in cmc_scores:
        # set all values that are empty before to None
        if pos is not None and not numpy.array(pos).size:
            pos = None
        if neg is not None and not numpy.array(neg).size:
            neg = None

        if pos is None and neg is None:
            raise ValueError(
                "One pair of the CMC scores has neither positive nor negative values"
            )

        # filter out any negative or positive scores below threshold; scores with exactly the threshold are also filtered out
        # now, None and an empty array have different meanings.
        if threshold is not None:
            if neg is not None:
                neg = numpy.array(neg)[neg > threshold]
            if pos is not None:
                pos = numpy.array(pos)[pos > threshold]

        if pos is None:
            # no positives, so we definitely do not have a match;
            # check if we have negatives above threshold
            if not neg.size:
                # we have no negative scores over the threshold, so we have correctly rejected the probe
                # don't increase any of the two counters...
                continue
            # we have negatives over threshold, so we have incorrect classifications; independent on the actual rank
            counter += 1
        else:
            # we have a positive, so we need to count the probe
            counter += 1

            if not numpy.array(pos).size:
                # all positive scores have been filtered out by the threshold, we definitely have a mis-match
                continue

            # get the maximum positive score for the current probe item
            # (usually, there is only one positive score, but just in case...)
            max_pos = numpy.max(pos)

            if neg is None or not numpy.array(neg).size:
                # if we had no negatives, or all negatives were below threshold, we have a match at rank 1
                correct += 1
            else:
                # count the number of negative scores that are higher than the best positive score
                index = numpy.sum(neg >= max_pos)
                if index < rank:
                    correct += 1

    return float(correct) / float(counter)


def cmc(cmc_scores):
    """Calculates the cumulative match characteristic (CMC) from the given input.

    The input has a specific format, which is a list of two-element tuples. Each
    of the tuples contains the negative and the positive scores for one probe
    item.

    For each probe item the probability that the rank :math:`r` of the positive
    score is calculated.  The rank is computed as the number of negative scores
    that are higher than the positive score.  If several positive scores for one
    test item exist, the **highest** positive score is taken. The CMC finally
    computes how many test items have rank r or higher, divided by the total
    number of test values.

    .. note::

       The CMC is not available for open set classification. Please use the
       :py:func:`detection_identification_rate` and :py:func:`false_alarm_rate`
       instead.


    Parameters
    ----------

    cmc_scores : :py:class:`list`
      A list in the format ``[(negatives, positives), ...]`` containing the CMC
      scores.

      Each pair contains the ``negative`` and the ``positive`` scores for **one
      probe item**.  Each pair can contain up to one empty array (or ``None``),
      i.e., in case of open set recognition.


    Returns
    -------

    1D :py:class:`numpy.ndarray` of `float`
      A 1D float array representing the CMC curve.
      The rank 1 recognition rate can be found in ``array[0]``, rank 2 rate in
      ``array[1]``, and so on. The number of ranks (``array.shape[0]``) is the
      number of gallery items. Values are in range ``[0,1]``.
    """

    # If no scores are given, we cannot plot anything
    probe_count = float(len(cmc_scores))
    if not probe_count:
        raise ValueError("The given set of scores is empty")

    # compute MC
    match_characteristic = numpy.zeros(
        (max([len(neg) for neg, _ in cmc_scores if neg is not None]) + 1,),
        numpy.int,
    )

    for neg, pos in cmc_scores:
        if pos is None or not numpy.array(pos).size:
            raise ValueError(
                "For the CMC computation at least one positive score per pair is necessary."
            )
        if neg is None:
            neg = []

        # get the maximum positive score for the current probe item
        # (usually, there is only one positive score, but just in case...)
        max_pos = numpy.max(pos)

        # count the number of negative scores that are higher than the best positive score
        index = numpy.sum(neg >= max_pos)
        match_characteristic[index] += 1

    # cumulate
    cumulative_match_characteristic = numpy.cumsum(
        match_characteristic, dtype=numpy.float64
    )
    return cumulative_match_characteristic / probe_count


def detection_identification_rate(cmc_scores, threshold, rank=1):
    """Computes the `detection and identification rate` for the given threshold.

    This value is designed to be used in an open set identification protocol, and
    defined in Chapter 14.1 of [LiJain2005]_.

    Although the detection and identification rate is designed to be computed on
    an open set protocol, it uses only the probe elements, for which a
    corresponding gallery element exists.  For closed set identification
    protocols, this function is identical to :py:func:`recognition_rate`.  The
    only difference is that for this function, a ``threshold`` for the scores
    need to be defined, while for :py:func:`recognition_rate` it is optional.


    Parameters:

      cmc_scores (:py:class:`list`): A list in the format ``[(negatives,
        positives), ...]`` containing the CMC.

        Each pair contains the ``negative`` and the ``positive`` scores for **one
        probe item**.  Each pair can contain up to one empty array (or ``None``),
        i.e., in case of open set recognition.

      threshold (float): The decision threshold :math:`\\tau``.

      rank (:obj:`int`, optional): The rank for which the curve should be plotted


    Returns:

      float: The detection and identification rate for the given threshold.

    """

    # count the correctly classifier probes
    correct = 0
    counter = 0
    for neg, pos in cmc_scores:
        if pos is None or not numpy.array(pos).size:
            # we only consider probes with corresponding gallery items
            continue
        # we have an in-gallery probe
        counter += 1
        # check, if it is correctly classified
        if neg is None:
            neg = []

        # get the maximum positive score for the current probe item
        # (usually, there is only one positive score, but just in case...)
        max_pos = numpy.max(pos)

        index = numpy.sum(
            neg >= max_pos
        )  # compute the rank (in fact, rank - 1)
        if max_pos >= threshold and index < rank:
            correct += 1

    if not counter:
        logger.warn("No in-gallery probe was found")
        return 0.0

    return float(correct) / float(counter)


def false_alarm_rate(cmc_scores, threshold):
    """Computes the `false alarm rate` for the given threshold,.

    This value is designed to be used in an open set identification protocol, and
    defined in Chapter 14.1 of [LiJain2005]_.

    The false alarm rate is designed to be computed on an open set protocol, it
    uses only the probe elements, for which **no** corresponding gallery element
    exists.


    Parameters:

      cmc_scores (:py:class:`list`): A list in the format ``[(negatives,
        positives), ...]`` containing the CMC scores (i.e. :py:class:`list`:
        A list of tuples, where each tuple contains the
        ``negative`` and ``positive`` scores for one probe of the database).

        Each pair contains the ``negative`` and the ``positive`` scores for **one
        probe item**.  Each pair can contain up to one empty array (or ``None``),
        i.e., in case of open set recognition.

      threshold (float): The decision threshold :math:`\\tau``.


    Returns:

      float: The false alarm rate.

    """
    incorrect = 0
    counter = 0
    for neg, pos in cmc_scores:
        # we only consider the out-of-gallery probes, i.e., with no positive scores
        if pos is None or not numpy.array(pos).size:
            counter += 1

            # check if the probe is above threshold
            if neg is None or not numpy.array(neg).size:
                raise ValueError(
                    "One pair of the CMC scores has neither positive nor negative values"
                )
            if numpy.max(neg) >= threshold:
                incorrect += 1

    if not counter:
        logger.warn("No out-of-gallery probe was found")
        return 0.0

    return float(incorrect) / float(counter)


def eer(negatives, positives, is_sorted=False, also_farfrr=False):
    """Calculates the Equal Error Rate (EER).

    Please note that it is possible that eer != fpr != fnr.
    This function returns (fpr + fnr) / 2 as eer.
    If you also need the fpr and fnr values, set ``also_farfrr`` to ``True``.

    Parameters
    ----------
    negatives : ``array_like (1D, float)``
        The scores for comparisons of objects of different classes.
    positives : ``array_like (1D, float)``
        The scores for comparisons of objects of the same class.
    is_sorted : bool
        Are both sets of scores already in ascendantly sorted order?
    also_farfrr : bool
        If True, it will also return far and frr.

    Returns
    -------
    eer : float
        The Equal Error Rate (EER).
    fpr : float
        The False Positive Rate (FPR). Returned only when ``also_farfrr`` is
        ``True``.
    fnr : float
        The False Negative Rate (FNR). Returned only when ``also_farfrr`` is
        ``True``.
    """
    threshold = eer_threshold(negatives, positives, is_sorted)
    far, frr = farfrr(negatives, positives, threshold)
    if also_farfrr:
        return (far + frr) / 2.0, far, frr
    return (far + frr) / 2.0


def roc_auc_score(
    negatives, positives, npoints=2000, min_far=-8, log_scale=False
):
    """Area Under the ROC Curve.
    Computes the area under the ROC curve. This is useful when you want to report one
    number that represents an ROC curve. This implementation uses the trapezoidal rule for
    the integration of the ROC curve. For more information, see:
    https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve


    Parameters
    ----------
    negatives : array_like
        The negative scores.
    positives : array_like
        The positive scores.
    npoints : int, optional
        Number of points in the ROC curve. Higher numbers leads to more accurate ROC.
    min_far : float, optional
        Min FAR and FRR values to consider when calculating ROC.
    log_scale : bool, optional
        If True, converts the x axis (FPR) to log10 scale before calculating AUC. This is
        useful in cases where len(negatives) >> len(positives)

    Returns
    -------
    float
        The ROC AUC. If ``log_scale`` is False, the value should be between 0 and 1.
    """
    fpr, fnr = roc(negatives, positives, npoints, min_far=min_far)
    tpr = 1 - fnr

    if log_scale:
        fpr_pos = fpr > 0
        fpr, tpr = fpr[fpr_pos], tpr[fpr_pos]
        fpr = numpy.log10(fpr)

    area = -1 * numpy.trapz(tpr, fpr)
    return area
