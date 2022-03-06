''' utility functions for bob.measure '''

import numpy
import scipy.stats
import logging

LOGGER = logging.getLogger(__name__)


def remove_nan(scores):
    """remove_nan

    Remove NaN(s) in the given array

    Parameters
    ----------
    scores :
        :py:class:`numpy.ndarray` : array

    Returns
    -------
        :py:class:`numpy.ndarray` : array without NaN(s)
        :py:class:`int` : number of NaN(s) in the input array
        :py:class:`int` : length of the input array
    """
    nans = numpy.isnan(scores)
    sum_nans = sum(nans)
    total = len(scores)
    if sum_nans > 0:
        LOGGER.warning('Found {} NaNs in {} scores'.format(sum_nans, total))
    return scores[~nans], sum_nans, total


def get_fta(scores):
    """get_fta
        calculates the Failure To Acquire (FtA) rate, i.e. proportion of NaN(s)
        in the input scores

    Parameters
    ----------
    scores :
        Tuple of (``positive``, ``negative``) :py:class:`numpy.ndarray`.

    Returns
    -------
    (:py:class:`numpy.ndarray`, :py:class:`numpy.ndarray`): scores without
    NaN(s)
    :py:class:`float` : failure to acquire rate
    """
    fta_sum, fta_total = 0.0, 0.0
    neg, sum_nans, total = remove_nan(scores[0])
    fta_sum += sum_nans
    fta_total += total
    pos, sum_nans, total = remove_nan(scores[1])
    fta_sum += sum_nans
    fta_total += total
    return ((neg, pos), fta_sum / fta_total)


def get_fta_list(scores):
    """ Get FTAs for a list of scores

    Parameters
    ----------
    scores: :any:`list`
        list of scores

    Returns
    -------
    neg_list: :any:`list`
        list of negatives
    pos_list: :any:`list`
        list of positives
    fta_list: :any:`list`
        list of FTAs
    """
    neg_list = []
    pos_list = []
    fta_list = []
    for score in scores:
        neg = pos = fta = None
        if score is not None:
            (neg, pos), fta = get_fta(score)
            if neg is None:
                raise ValueError("While loading dev-score file")
        neg_list.append(neg)
        pos_list.append(pos)
        fta_list.append(fta)
    return (neg_list, pos_list, fta_list)


def get_thres(criter, neg, pos, far=None):
    """Get threshold for the given positive/negatives scores and criterion

    Parameters
    ----------
    criter :
        Criterion (`eer` or `hter` or `far`)
    neg : :py:class:`numpy.ndarray`:
        array of negative scores
        pos : :py:class:`numpy.ndarray`::
        array of positive scores

    Returns
    -------
    :py:obj:`float`
        threshold
    """
    if criter == 'eer':
        from .brute_force import eer_threshold
        return eer_threshold(neg, pos)
    elif criter == 'min-hter':
        from .brute_force import min_hter_threshold
        return min_hter_threshold(neg, pos)
    elif criter == 'far':
        if far is None:
            raise ValueError("FAR value must be provided through "
                             "``--far-value`` or ``--fpr-value`` option.")
        from .brute_force import far_threshold
        return far_threshold(neg, pos, far)
    else:
        raise ValueError("Incorrect plotting criterion: ``%s``" % criter)


def get_colors(n):
    """get_colors
    Get a list of matplotlib colors

    Parameters
    ----------
    n : :obj:`int`
        Number of colors to output

    Returns
    -------
    :any:`list`
        list of colors
    """
    if n > 10:
        from matplotlib import pyplot
        cmap = pyplot.cm.get_cmap(name='magma')
        return [cmap(i) for i in numpy.linspace(0, 1.0, n + 1)]

    return ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']


def get_linestyles(n, on=True):
    """Get a list of matplotlib linestyles

    Parameters
    ----------
    n : :obj:`int`
        Number of linestyles to output

    Returns
    -------
    :any:`list`
        list of linestyles
    """
    if not on:
        return [None] * n

    list_linestyles = [
        (0, ()),  # solid
        (0, (1, 1)),  # densely dotted
        (0, (5, 5)),  # dashed
        (0, (5, 1)),  # densely dashed
        (0, (3, 1, 1, 1, 1, 1)),  # densely dashdotdotted
        (0, (3, 10, 1, 10, 1, 10)),  # loosely dashdotdotted
        (0, (3, 5, 1, 5, 1, 5)),  # dashdotdotted
        (0, (3, 1, 1, 1)),  # densely dashdotted
        (0, (1, 5)),  # dotted
        (0, (3, 5, 1, 5)),  # dashdotted
        (0, (5, 10)),  # loosely dashed
        (0, (3, 10, 1, 10)),  # loosely dashdotted
        (0, (1, 10))  # loosely dotted
    ]
    while n > len(list_linestyles):
        list_linestyles += list_linestyles
    return list_linestyles
