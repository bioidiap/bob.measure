''' Utilities functionalities'''

import scipy.stats
import numpy


def confidence_for_indicator_variable(x, n, alpha=0.05):
    '''Calculates the confidence interval for proportion estimates
    The Clopper-Pearson interval method is used for estimating the confidence
    intervals.

    More info on confidence intervals
    ---------------------------------
    https://en.wikipedia.org/wiki/Confidence_interval
    https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Clopper-Pearson_interval

    Parameters
    ----------
    x : int
        The number of successes.
    n : int
        The number of trials.
    alpha : float, optional
        The 1-confidence value that you want. For example, alpha should be 0.05
        to obtain 95% confidence intervals.

    Returns
    -------
    (float, float) Returns a tuple of (lower_bound, upper_bound) which
        shows the limit of your success rate: lower_bound < x/n < upper_bound
    '''
    lower_bound = scipy.stats.beta.ppf(alpha / 2.0, x, n - x + 1)
    upper_bound = scipy.stats.beta.ppf(1 - alpha / 2.0, x + 1, n - x)
    if numpy.isnan(lower_bound):
        lower_bound = 0
    if numpy.isnan(upper_bound):
        upper_bound = 1
    return (lower_bound, upper_bound)
