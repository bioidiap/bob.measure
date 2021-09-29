#!/usr/bin/env python
# coding=utf-8

"""Frequentist confidence interval estimation

(Frequentist) confidence interval interpretation, with 95% coverage: **If we
are to take several independent random samples from the population and
construct confidence intervals from each of the sample data, then 95 out of 100
confidence intervals will contain the true mean (true proportion, in this
context of proportion)**.

See a discussion in `Five Confidence Intervals for Proportions That You
Should Know About <ci-evaluation_>`_.

.. include:: ../links.rst
"""

import numbers
import numpy
import scipy.stats


def clopper_pearson(k, l, coverage=0.95):
    """Calculates the "exact" confidence interval for proportion estimates

    The Clopper-Pearson interval method is used for estimating the confidence
    intervals.  This implementation is based on [CLOPPER-1934]_.  This
    technique is **very** conservative - in most of the cases, coverage is
    greater than the required value, which may imply in too large confidence
    intervals.


    Parameters
    ==========

    k : int, numpy.ndarray (int, 1D)
        Number of successes observed on the experiment

    l : int, numpy.ndarray (int, 1D)
        Number of failures observed on the experiment

    coverage : :py:class:`float`, Optional
        A floating-point number between 0 and 1.0 indicating the
        coverage you're expecting.  A value of 0.95 will ensure 95% coverage.


    Returns
    -------

    lower, upper: float, numpy.ndarray (float, 1D)
        The lower and upper bounds of the credible region

    """
    is_scalar = isinstance(k, numbers.Integral)
    if is_scalar:
        # make it an array
        k = numpy.array(k).reshape((1,))
        l = numpy.array(l).reshape((1,))

    right = (1.0 - coverage) / 2  # half-width in each side
    lower = scipy.stats.beta.ppf(right, k, l + 1)
    upper = scipy.stats.beta.ppf(1 - right, k + 1, l)

    lower[numpy.isnan(lower)] = 0.0
    upper[numpy.isnan(upper)] = 1.0

    if is_scalar:
        return lower[0], upper[0]
    else:
        return lower, upper


def agresti_coull(k, l, coverage=0.95):
    """Calculates the confidence interval for proportion estimates

    The Agresti-Coull interval method is used for estimating the confidence
    intervals.  This implementation is based on [AGRESTI-1998]_.  This
    technique is conservative - in most of the cases, coverage is greater
    than the required value, which may imply a larger confidence interval that
    required.

    This function is considered a good choice for the frequentist approach, if
    you cannot use :py:func:`clopper_pearson`.


    Parameters
    ==========

    k : int, numpy.ndarray (int, 1D)
        Number of successes observed on the experiment

    l : int, numpy.ndarray (int, 1D)
        Number of failures observed on the experiment

    coverage : :py:class:`float`, Optional
        A floating-point number between 0 and 1.0 indicating the
        coverage you're expecting.  A value of 0.95 will ensure 95% coverage.


    Returns
    -------

    lower, upper: float, numpy.ndarray (float, 1D)
        The lower and upper bounds of the credible region

    """
    is_scalar = isinstance(k, numbers.Integral)
    if is_scalar:
        # make it an array
        k = numpy.array(k).reshape((1,))
        l = numpy.array(l).reshape((1,))

    right = (1.0 - coverage) / 2  # half-width in each side
    crit = scipy.stats.norm.isf(right)
    kl_c = (k + l) + crit ** 2
    q_c = (k + crit ** 2 / 2.0) / kl_c
    std_c = numpy.sqrt(q_c * (1.0 - q_c) / kl_c)
    dist = crit * std_c
    lower = q_c - dist
    upper = q_c + dist

    lower[numpy.isnan(lower)] = 0.0
    upper[numpy.isnan(upper)] = 1.0

    if is_scalar:
        return lower[0], upper[0]
    else:
        return lower, upper


def wilson(k, l, coverage=0.95):
    """Calculates the confidence interval for proportion estimates

    The Wilson interval method is used for estimating the confidence intervals.
    This implementation is based on [WILSON-1927]_.  This implementation does
    **not** contain the continuity correction.  It is as conservative in the
    extremes of the domain as the bayesian approach and can be a good default,
    if :py:func:`clopper_pearson` cannot be used.

    This function is considered the best "default" for the frequentist
    approach as it is not too conservative and assumes a resonable value
    through out the range.


    Parameters
    ==========

    k : int, numpy.ndarray (int, 1D)
        Number of successes observed on the experiment

    l : int, numpy.ndarray (int, 1D)
        Number of failures observed on the experiment

    coverage : :py:class:`float`, Optional
        A floating-point number between 0 and 1.0 indicating the
        coverage you're expecting.  A value of 0.95 will ensure 95% coverage.


    Returns
    -------

    lower, upper: float, numpy.ndarray (float, 1D)
        The lower and upper bounds of the credible region

    """
    is_scalar = isinstance(k, numbers.Integral)
    if is_scalar:
        # make it an array
        k = numpy.array(k).reshape((1,))
        l = numpy.array(l).reshape((1,))

    right = (1.0 - coverage) / 2  # half-width in each side
    n = k + l
    p = k / n
    crit = scipy.stats.norm.isf(right)
    crit2 = crit ** 2
    denom = 1 + crit2 / n
    center = (p + crit2 / (2 * n)) / denom
    dist = crit * numpy.sqrt(p * (1.0 - p) / n + crit2 / (4.0 * n ** 2))
    dist /= denom
    lower = center - dist
    upper = center + dist

    lower[numpy.isnan(lower)] = 0.0
    upper[numpy.isnan(upper)] = 1.0

    if is_scalar:
        return lower[0], upper[0]
    else:
        return lower, upper
