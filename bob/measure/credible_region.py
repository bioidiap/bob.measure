#!/usr/bin/env python
# coding=utf-8

"""Functions to evalute the (Bayesian) credible region of measures

(Bayesian) credible region interpretation, with 95% coverage: The probability
that the true proportion will lie within the 95% credible interval is 0.95.

Contrary to frequentist approaches, in which one can only say that if the test
were repeated an infinite number of times, and one constructed a confidence
interval each time, then X% of the confidence intervals would contain the true
rate, here we can say that given our observed data, there is a X% probability
that the true value of :math:`k/n` falls within the provided interval.

See a discussion in `Five Confidence Intervals for Proportions That You
Should Know About <ci-evaluation_>`_ for a study on coverage for most common
methods.

.. note::

   For a disambiguation with `Confidence Interval <confidence-interval_>`_ (the
   frequentist approach), read `Credible Regions or Intervals
   <credible-interval_>`_.

.. include:: ../links.rst
"""

import numbers
import numpy
import numpy.random
import scipy.special


def beta(k, l, lambda_, coverage):
    """
    Returns the mode, upper and lower bounds of the equal-tailed credible
    region of a probability estimate following Bernoulli trials.

    This technique is (not) very conservative - in most of the cases, coverage
    closer to the extremes (0 or 1) is lower than expected (but still greater
    than 85%).

    This implementation is based on [GOUTTE-2005]_.  It assumes :math:`k`
    successes and :math:`l` failures (:math:`n = k+l` total trials) are issued
    from a series of Bernoulli trials (likelihood is binomial).  The posterior
    is derivated using the Bayes Theorem with a beta prior.  As there is no
    reason to favour high vs.  low precision, we use a symmetric Beta prior
    (:math:`\\alpha=\\beta`):

    .. math::

       P(p|k,n) &= \\frac{P(k,n|p)P(p)}{P(k,n)} \\\\
       P(p|k,n) &= \\frac{\\frac{n!}{k!(n-k)!}p^{k}(1-p)^{n-k}P(p)}{P(k)} \\\\
       P(p|k,n) &= \\frac{1}{B(k+\\alpha, n-k+\beta)}p^{k+\\alpha-1}(1-p)^{n-k+\\beta-1} \\\\
       P(p|k,n) &= \\frac{1}{B(k+\\alpha, n-k+\\alpha)}p^{k+\\alpha-1}(1-p)^{n-k+\\alpha-1}

    The mode for this posterior (also the maximum a posteriori) is:

    .. math::

       \\text{mode}(p) = \\frac{k+\\lambda-1}{n+2\\lambda-2}

    Concretely, the prior may be flat (all rates are equally likely,
    :math:`\\lambda=1`) or we may use Jeoffrey's prior (:math:`\\lambda=0.5`),
    that is invariant through re-parameterisation.  Jeffrey's prior indicate
    that rates close to zero or one are more likely.

    The mode above works if :math:`k+{\\alpha},n-k+{\\alpha} > 1`, which is
    usually the case for a resonably well tunned system, with more than a few
    samples for analysis.  In the limit of the system performance, :math:`k`
    may be 0, which will make the mode become zero.

    For our purposes, it may be more suitable to represent :math:`n = k + l`,
    with :math:`k`, the number of successes and :math:`l`, the number of
    failures in the binomial experiment, and find this more suitable
    representation:

    .. math::

       P(p|k,l) &= \\frac{1}{B(k+\\alpha, l+\\alpha)}p^{k+\\alpha-1}(1-p)^{l+\\alpha-1} \\\\
       \\text{mode}(p) &= \\frac{k+\\lambda-1}{k+l+2\\lambda-2}

    This can be mapped to most rates calculated in the context of binary
    classification this way:

    * Precision or Positive-Predictive Value (PPV): p = TP/(TP+FP), so k=TP, l=FP
    * Recall, Sensitivity, or True Positive Rate: r = TP/(TP+FN), so k=TP, l=FN
    * Specificity or True Negative Rage: s = TN/(TN+FP), so k=TN, l=FP
    * F1-score: f1 = 2TP/(2TP+FP+FN), so k=2TP, l=FP+FN
    * Accuracy: acc = TP+TN/(TP+TN+FP+FN), so k=TP+TN, l=FP+FN
    * Jaccard: j = TP/(TP+FP+FN), so k=TP, l=FP+FN


    Parameters
    ==========

    k : int, numpy.ndarray (int, 1D)
        Number of successes observed on the experiment

    l : int, numpy.ndarray (int, 1D)
        Number of failures observed on the experiment

    lambda__ : :py:class:`float`, Optional
        The parameterisation of the Beta prior to consider. Use
        :math:`\\lambda=1` for a flat prior.  Use :math:`\\lambda=0.5` for
        Jeffrey's prior (the default).

    coverage : :py:class:`float`, Optional
        A floating-point number between 0 and 1.0 indicating the
        coverage you're expecting.  A value of 0.95 will ensure 95%
        of the area under the probability density of the posterior
        is covered by the returned equal-tailed interval.


    Returns
    =======

    mean : float, numpy.ndarray (float, 1D)
        The mean of the posterior distribution

    mode : float, numpy.ndarray (float, 1D)
        The mode of the posterior distribution

    lower, upper: float, numpy.ndarray (float, 1D)
        The lower and upper bounds of the credible region

    """
    is_scalar = isinstance(k, numbers.Integral)
    if is_scalar:
        # make it an array
        k = numpy.array(k).reshape((1,))
        l = numpy.array(l).reshape((1,))

    # we return the equally-tailed range
    right = (1.0 - coverage) / 2  # half-width in each side
    lower = scipy.special.betaincinv(k + lambda_, l + lambda_, right)
    upper = scipy.special.betaincinv(k + lambda_, l + lambda_, 1.0 - right)

    # evaluate mean and mode (https://en.wikipedia.org/wiki/Beta_distribution)
    alpha = k + lambda_
    beta = l + lambda_

    E = alpha / (alpha + beta)

    # the mode of a beta distribution is a bit tricky
    mode = numpy.zeros_like(lower)
    cond = (alpha > 1) & (beta > 1)
    mode[cond] = (alpha[cond] - 1) / (alpha[cond] + beta[cond] - 2)
    # In the case of precision, if the threshold is close to 1.0, both TP
    # and FP can be zero, which may cause this condition to be reached, if
    # the prior is exactly 1 (flat prior).  This is a weird situation,
    # because effectively we are trying to compute the posterior when the
    # total number of experiments is zero.  So, only the prior counts - but
    # the prior is flat, so we should just pick a value.  We choose the
    # middle of the range.
    # conda = alpha == 1 and beta == 1
    # mode[cond] = 0.0
    # conda = alpha <= 1 and beta > 1
    # mode[cond] = 0.0
    mode[(alpha > 1) & (beta <= 1)] = 1.0
    # else: #elif alpha < 1 and beta < 1:
    # in the case of precision, if the threshold is close to 1.0, both TP
    # and FP can be zero, which may cause this condition to be reached, if
    # the prior is smaller than 1.  This is a weird situation, because
    # effectively we are trying to compute the posterior when the total
    # number of experiments is zero.  So, only the prior counts - but the
    # prior is bimodal, so we should just pick a value.  We choose the
    # left of the range.
    # n.b.: could also be 1.0 as the prior is bimodal
    # mode[alpha < 1 and beta < 1] = 0.0

    if is_scalar:
        return E[0], mode[0], lower[0], upper[0]
    else:
        return E, mode, lower, upper


def beta_posterior(k, l, lambda_, nb_samples):
    """Simulates the beta posterior of a system with the provided markings

    This implementation is based on [GOUTTE-2005]_, equation 7.

    Figures of merit that are supported by this procedure are those which have
    the form :math:`v = k / (k + l)`:

    * Precision or Positive-Predictive Value (PPV): :math:`p = TP/(TP+FP)`, so
      :math:`k=TP`, :math:`l=FP`
    * Recall, Sensitivity, or True Positive Rate: :math:`r = TP/(TP+FN)`, so
      :math:`k=TP`, :math:`l=FN`
    * Specificity or True Negative Rage: :math:`s = TN/(TN+FP)`, so :math:`k=TN`,
      :math:`l=FP`
    * Accuracy: :math:`acc = TP+TN/(TP+TN+FP+FN)`, so :math:`k=TP+TN`,
      :math:`l=FP+FN`
    * Jaccard Index: :math:`j = TP/(TP+FP+FN)`, so :math:`k=TP`, :math:`l=FP+FN`

    Parameters
    ----------

    k : int
        Depends on the figure of merit being considered (see above)

    l : int
        Depends on the figure of merit being considered (see above)

    lambda_ : float
        The parameterisation of the Beta prior to consider. Use
        :math:`\lambda=1` for a flat prior.  Use :math:`\lambda=0.5` for
        Jeffrey's prior.

    nb_samples : int
        number of generated gamma distribution values


    Returns
    -------

    variates : numpy.ndarray
        An array with size ``nb_samples`` containing a realization of equation 7.

    """

    return numpy.random.beta(a=(k + lambda_), b=(l + lambda_), size=nb_samples)


def compare_beta_posteriors(k1, l1, k2, l2, lambda_, nb_samples):
    """
    Returns the probability that system 1 is better than system 2 for a given
    figure of merit.

    This implementation is based on [GOUTTE-2005]_.

    Figures of merit that are supported by this procedure are those which have
    the form :math:`v = k / (k + l)`:

    * Precision or Positive-Predictive Value (PPV): :math:`p = TP/(TP+FP)`, so
      :math:`k=TP`, :math:`l=FP`
    * Recall, Sensitivity, or True Positive Rate: :math:`r = TP/(TP+FN)`, so
      :math:`k=TP`, :math:`l=FN`
    * Specificity or True Negative Rage: :math:`s = TN/(TN+FP)`, so :math:`k=TN`,
      :math:`l=FP`
    * Accuracy: :math:`acc = TP+TN/(TP+TN+FP+FN)`, so :math:`k=TP+TN`,
      :math:`l=FP+FN`
    * Jaccard Index: :math:`j = TP/(TP+FP+FN)`, so :math:`k=TP`, :math:`l=FP+FN`


    Parameters
    ----------

    k1/2 : int
        Depends on the figure of merit being considered (see above)

    l1/2 : int
        Depends on the figure of merit being considered (see above)

    lambda_ : float
        The parameterisation of the Beta prior to consider. Use
        :math:`\lambda=1` for a flat prior.  Use :math:`\lambda=0.5` for
        Jeffrey's prior.

    nb_samples : int
        number of generated beta distribution values


    Returns
    -------

    prob : float
        A number between 0.0 and 1.0 that describes the probability that the
        first system has a bigger measurement than the second

    """

    v1 = beta_posterior(k1, l1, lambda_, nb_samples)
    v2 = beta_posterior(k2, l2, lambda_, nb_samples)
    return numpy.count_nonzero(v1 > v2) / nb_samples


def f1_posterior(tp, fp, fn, lambda_, nb_samples):
    """Simulates the F1-score posterior of a system with the provided markings

    This implementation is based on [GOUTTE-2005]_, equation 11.

    Parameters
    ----------

    tp : int
        True positive count, AKA "hit"

    fp : int
        False positive count, AKA "false alarm", or "Type I error"

    fn : int
        False Negative count, AKA "miss", or "Type II error"

    lambda_ : float
        The parameterisation of the Beta prior to consider. Use
        :math:`\lambda=1` for a flat prior.  Use :math:`\lambda=0.5` for
        Jeffrey's prior.

    nb_samples : int
        number of generated gamma distribution values


    Returns
    -------

    variates : numpy.ndarray
        An array with size ``nb_samples`` containing a realization of equation
        11.

    """

    u = numpy.random.gamma(shape=(tp + lambda_), scale=2.0, size=nb_samples)
    v = numpy.random.gamma(
        shape=(fp + fn + (2 * lambda_)), scale=1.0, size=nb_samples
    )
    return u / (u + v)


def compare_f1_scores(tp1, fp1, fn1, tp2, fp2, fn2, lambda_, nb_samples):
    """
    Returns the probability that the F1-score from 1 system is bigger than the
    F1-score of a second system.

    This implementation is based on [GOUTTE-2005]_.


    Parameters
    ----------

    tp1/2 : int
        True positive count, AKA "hit"

    fp1/2 : int
        False positive count, AKA "false alarm", or "Type I error"

    fn1/2 : int
        False Negative count, AKA "miss", or "Type II error"

    lambda_ : float
        The parameterisation of the Beta prior to consider. Use
        :math:`\lambda=1` for a flat prior.  Use :math:`\lambda=0.5` for
        Jeffrey's prior.

    nb_samples : int
        number of generated gamma distribution values


    Returns
    -------

    prob : float
        A number between 0.0 and 1.0 that describes the probability that the
        first system has a bigger F1-score than the second

    """

    f1 = f1_posterior(tp1, fp1, fn1, lambda_, nb_samples)
    f2 = f1_posterior(tp2, fp2, fn2, lambda_, nb_samples)
    return numpy.count_nonzero(f1 > f2) / nb_samples


def f1_score(tp, fp, fn, lambda_, coverage, nb_samples):
    """
    Returns the mean, mode, upper and lower bounds of the credible
    region of the F1 score.

    This implementation is based on [GOUTTE-2005]_.


    Parameters
    ----------

    tp : int
        True positive count, AKA "hit"

    fp : int
        False positive count, AKA "false alarm", or "Type I error"

    fn : int
        False Negative count, AKA "miss", or "Type II error"

    lambda_ : float
        The parameterisation of the Beta prior to consider. Use
        :math:`\lambda=1` for a flat prior.  Use :math:`\lambda=0.5` for
        Jeffrey's prior.

    coverage : float
        A floating-point number between 0 and 1.0 indicating the coverage
        you're expecting.  A value of 0.95 will ensure 95% of the area under
        the probability density of the posterior is covered by the returned
        equal-tailed interval.

    nb_samples : int
        number of generated gamma distribution values


    Returns
    -------

    f1_score : (float, float, float, float)
        F1, mean, mode and credible intervals (95% CI). See `F1-score
        <https://en.wikipedia.org/wiki/F1_score>`_.  It corresponds
        arithmetically to ``2*P*R/(P+R)`` or ``2*tp/(2*tp+fp+fn)``.  The F1 or
        Dice score depends on a TP-only numerator, similarly to the Jaccard
        index.  For regions where there are no annotations, the F1-score will
        always be zero, irrespective of the model output.  Accuracy may be a
        better proxy if one needs to consider the true abscence of annotations
        in a region as part of the measure.

    """

    scores = f1_posterior(tp, fp, fn, lambda_, nb_samples)

    left_half = (1 - coverage) / 2  # size of excluded (half) area
    sorted_scores = numpy.sort(scores)

    # n.b.: we return the equally tailed range

    # calculates position of score which would exclude the left_half (left)
    lower_index = int(round(nb_samples * left_half))

    # calculates position of score which would exclude the left_half (right)
    upper_index = int(round(nb_samples * (1 - left_half)))

    lower = sorted_scores[lower_index - 1]
    upper = sorted_scores[upper_index - 1]

    return numpy.mean(scores), scipy.stats.mode(scores)[0][0], lower, upper


def measures(tp, fp, tn, fn, lambda_, coverage):
    """Calculates mean and mode from true/false positive and negative counts
    with credible regions

    This function can return bayesian estimates of standard machine learning
    measures from true and false positive counts of positives and negatives.
    For a thorough look into these and alternate names for the returned values,
    please check Wikipedia's entry on `Precision and Recall
    <https://en.wikipedia.org/wiki/Precision_and_recall>`_.  See
    :py:func:`beta_credible_region` for details on the calculation of returned
    values.


    Parameters
    ----------

    tp : int
        True positive count, AKA "hit"

    fp : int
        False positive count, AKA "false alarm", or "Type I error"

    tn : int
        True negative count, AKA "correct rejection"

    fn : int
        False Negative count, AKA "miss", or "Type II error"

    lambda_ : float
        The parameterisation of the Beta prior to consider. Use
        :math:`\lambda=1` for a flat prior.  Use :math:`\lambda=0.5` for
        Jeffrey's prior.

    coverage : float
        A floating-point number between 0 and 1.0 indicating the coverage
        you're expecting.  A value of 0.95 will ensure 95% of the area under
        the probability density of the posterior is covered by the returned
        equal-tailed interval.



    Returns
    -------

    precision : (float, float, float, float)
        P, AKA positive predictive value (PPV), mean, mode and credible
        intervals (95% CI).  It corresponds arithmetically to ``tp/(tp+fp)``.

    recall : (float, float, float, float)
        R, AKA sensitivity, hit rate, or true positive rate (TPR), mean, mode
        and credible intervals (95% CI).  It corresponds arithmetically to
        ``tp/(tp+fn)``.

    specificity : (float, float, float, float)
        S, AKA selectivity or true negative rate (TNR), mean, mode and credible
        intervals (95% CI).  It corresponds arithmetically to ``tn/(tn+fp)``.

    accuracy : (float, float, float, float)
        A, mean, mode and credible intervals (95% CI).  See `Accuracy
        <https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers>`_. is
        the proportion of correct predictions (both true positives and true
        negatives) among the total number of pixels examined.  It corresponds
        arithmetically to ``(tp+tn)/(tp+tn+fp+fn)``.  This measure includes
        both true-negatives and positives in the numerator, what makes it
        sensitive to data or regions without annotations.

    jaccard : (float, float, float, float)
        J, mean, mode and credible intervals (95% CI).  See `Jaccard Index or
        Similarity <https://en.wikipedia.org/wiki/Jaccard_index>`_.  It
        corresponds arithmetically to ``tp/(tp+fp+fn)``.  The Jaccard index
        depends on a TP-only numerator, similarly to the F1 score.  For regions
        where there are no annotations, the Jaccard index will always be zero,
        irrespective of the model output.  Accuracy may be a better proxy if
        one needs to consider the true abscence of annotations in a region as
        part of the measure.

    f1_score : (float, float, float, float)
        F1, mean, mode and credible intervals (95% CI). See `F1-score
        <https://en.wikipedia.org/wiki/F1_score>`_.  It corresponds
        arithmetically to ``2*P*R/(P+R)`` or ``2*tp/(2*tp+fp+fn)``.  The F1 or
        Dice score depends on a TP-only numerator, similarly to the Jaccard
        index.  For regions where there are no annotations, the F1-score will
        always be zero, irrespective of the model output.  Accuracy may be a
        better proxy if one needs to consider the true abscence of annotations
        in a region as part of the measure.

    """

    return (
        beta(tp, fp, lambda_, coverage),  # precision
        beta(tp, fn, lambda_, coverage),  # recall
        beta(tn, fp, lambda_, coverage),  # specificity
        beta(tp + tn, fp + fn, lambda_, coverage),  # accuracy
        beta(tp, fp + fn, lambda_, coverage),  # jaccard index
        f1_score(tp, fp, fn, lambda_, coverage, 100000),  # f1-score
    )
