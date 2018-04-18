''' utility functions for bob.measure '''

import numpy
import scipy.stats
import bob.core

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
        logger = bob.core.log.setup("bob.measure")
        logger.warning('Found {} NaNs in {} scores'.format(sum_nans, total))
    return scores[numpy.where(~nans)], sum_nans, total

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
    fta_sum, fta_total = 0, 0
    neg, sum_nans, total = remove_nan(scores[0])
    fta_sum += sum_nans
    fta_total += total
    pos, sum_nans, total = remove_nan(scores[1])
    fta_sum += sum_nans
    fta_total += total
    return ((neg, pos), fta_sum / fta_total)

def get_thres(criter, neg, pos, far=None):
    """Get threshold for the given positive/negatives scores and criterion

    Parameters
    ----------
    criter :
        Criterion (`eer` or `hter`)
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
        from . import eer_threshold
        return eer_threshold(neg, pos)
    elif criter == 'hter':
        from . import min_hter_threshold
        return min_hter_threshold(neg, pos)
    elif criter == 'far':
        if far is None:
            raise ValueError("FAR value must be provided through "
                             "``--far-value`` option.")
        from . import far_threshold
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

    return ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']

def confidence_for_indicator_variable(x, n, alpha=0.05):
    '''Calculates the confidence interval for proportion estimates
    The Clopper-Pearson interval method is used for estimating the confidence
    intervals.

    Parameters
    ----------
    x : int
        The number of successes.
    n : int
        The number of trials.
        alpha : :obj:`float`, optional
        The 1-confidence value that you want. For example, alpha should be 0.05
        to obtain 95% confidence intervals.

    Returns
    -------
    (:obj:`float`, :obj:`float`)
        a tuple of (lower_bound, upper_bound) which
        shows the limit of your success rate: lower_bound < x/n < upper_bound
    '''
    lower_bound = scipy.stats.beta.ppf(alpha / 2.0, x, n - x + 1)
    upper_bound = scipy.stats.beta.ppf(1 - alpha / 2.0, x + 1, n - x)
    if numpy.isnan(lower_bound):
        lower_bound = 0
    if numpy.isnan(upper_bound):
        upper_bound = 1
    return (lower_bound, upper_bound)
