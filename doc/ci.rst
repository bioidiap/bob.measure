.. coding=utf-8

.. testsetup:: *

   from matplotlib import pyplot as plt
   import bob.measure

===================================
 Credible and Confidence Intervals
===================================


Credible Interval (or Region)
-----------------------------

A `Credible Interval <credible-interval_>`_ or region (for multi-dimensional
cases) for parameter :math:`x` consists of a lower estimate :math:`L`, and an
upper estimate :math:`U`, such that the probability of the true value being
within the interval estimate is equal to :math:`\alpha`.  For example, a 95%
credible interval (i.e.  :math:`\alpha = 0.95`) for a parameter :math:`x` is
given by :math:`[L, U]` such that

.. math::
   P(k \in [L,U]) = 95%

The smaller the test size, the wider the confidence interval will be, and the
greater :math:`\alpha`, the smaller the confidence interval will be.  The
evaluation of credible intervals follows a bayesian approach, where one assumes
a prior probability density function that models the likelihood of the
parameter, given its possible range.  Once the prior function is established,
the Bayes theorem is used to devise the posterior distribution of the
parameter given its current estimate, and to calculate the credible interval.


Background
==========

In binary classification problems **where each sample is i.i.d.** (independent
and identically distributed random variables), success/failure tests are
binomially distributed (that is, composed of a number :math:`n` of Bernoulli
trials using a biased coin with "unknown" proportion :math:`p`):

.. math::
   P (K = k\mid p) = \binom{n}{k} p^k (1-p)^{n-k}

where:

* :math:`n` is the total number of trials for a particular experiment,
* :math:`k` is the number of successes within those trials, and
* :math:`p` is the true probabibility, which is unknown, and that we are trying
  to estimate from experiments.

To estimate :math:`p`, the true value for the unknow probability, given
:math:`k`, the number of successes observed in a concrete experiment, we apply
the Bayes theorem:

.. math::
   P(p\mid k) = \frac{P(k\mid p) P(p)}{P(k)}

The value of :math:`P(k)` does not depend on :math:`n` and can be recast as a
marginalized version of :math:`P(k\mid p)`, which will help us later:

.. math::
   P(p\mid k) &= \frac{P(k\mid p) P(p)} {P(k)} \\
              & = \frac{P(k\mid p) P(p)} {\int_{p'} P(k, p') dp'} \\
              & = \frac{P(k\mid p) P(p)} {\int_{p'} P(k\mid p') P(p') dp'}

In the case of a binomial distribution, :math:`P(k\mid p)` is known (first
equation).  :math:`P(k)` is normally called the *prior* probability density,
and corresponds to a known (or most likely) density distribution for the
parameter :math:`k`.  The choice of this prior will of course affect the
overall aspect of the posterior distribution :math:`P(p\mid k)` we are trying
to estimate.

A typical choice for this prior is a `Beta distribution`_.  As it turns out, a
Beta prior will generate a (conjugate) Beta posterior:

.. math::
   P(p\mid k) = \frac{1}{B(k+\alpha,n-k+\beta} p^{k+\alpha-1}(1-p)^{n-k+\beta-1}

This formulation provides us with a complete representation for the posterior
of :math:`p`, allowing the calculation of credible intervals, via integration.
The values :math:`\alpha` and :math:`\beta` are hyper-parameters which control
the skewness of the Beta distribution towards the maximum or the minimum
respectively.  As there is no reason to favour one more than the other, these
are typically set to matching values :math:`\alpha=\beta=\lambda`.

.. math::
   P(p\mid k) = \frac{1}{B(k+\lambda,n-k+\lambda} p^{k+\lambda-1}(1-p)^{n-k+\lambda-1}

Two classical settings are often used:

* :math:`\lambda = 1` (a.k.a. a "flat" prior)
* :math:`\lambda = 0.5` (a.k.a. Jeoffrey's prior)

In practice, changing :math:`\lambda` does not affect much the credible
interval calculation.  To calculate the credible interval for a binary variable
using a flat or Jeoffrey's prior, use
:py:func:`bob.measure.credible_region.beta`, providing :math:`k`,
:math:`l=(n-k)`, :math:`\lambda` and how much coverage you would like to have
(typically 0.95 - 95%).


Applicability to Figures of Merit in Classification
===================================================

[GOUTTE-2005]_ extended this analysis to classifical figures of merit in
classification, with a similar reasoning as used for accuracy on the previous
section:

* Precision or Positive-Predictive Value (PPV): :math:`p = TP/(TP+FP)`, so
  :math:`k=TP`, :math:`l=FP`
* Recall, Sensitivity, or True Positive Rate: :math:`r = TP/(TP+FN)`, so
  :math:`k=TP`, :math:`l=FN`
* Specificity or True Negative Rage: :math:`s = TN/(TN+FP)`, so :math:`k=TN`,
  :math:`l=FP`
* F1-score: :math:`f1 = 2TP/(2TP+FP+FN)`, so :math:`k=2TP`, :math:`l=FP+FN`
* Accuracy: :math:`acc = TP+TN/(TP+TN+FP+FN)`, so :math:`k=TP+TN`,
  :math:`l=FP+FN`
* Jaccard Index: :math:`j = TP/(TP+FP+FN)`, so :math:`k=TP`, :math:`l=FP+FN`

The function :py:func:`bob.measure.credible_region.measures` can calculate the
above quantites in a single shot from counts of true and false, positives and
negatives, the :math:`\lambda` parameter, and the desired coverage.


Confidence Interval
-------------------

A `confidence interval`_ corresponds to a *frequentist* approach to the
estimation of a range of values for an unknown parameter :math:`p`.  More
formally, a 95% confidence interval means that with a large number of `n`
repeated Bernoulli trials, 95% of times, the estimated interval would include
the true value of the parameter.  Where as in the Bayesian approach the
interval is fixed and the parameter :math:`p` is subject to random process, in
the frequentist approach, the parameter :math:`p` is fixed and the interval is
subject to randomness.

There are several proposed ways to calculate a confidence interval, some of
which are implemented in this package.  They differ by the "conservativeness",
or how large the interval will be for the same coverage.  Except for specific
method parameterization (:math:`\lambda`), they should (almost) work as a
drop-in replacement for :py:func:`bob.measure.credible_region.beta`:

* Wilson, 1927, [WILSON-1927]_:
  :py:func:`bob.measure.confidence_interval.wilson`
* Clopper-Pearson, 1934, [CLOPPER-1934]_:
  :py:func:`bob.measure.confidence_interval.clopper_pearson`
* Agresti-Coull, 1998, [AGRESTI-1998]_:
  :py:func:`bob.measure.confidence_interval.agresti_coull`


Conservativeness
----------------

When talking about credible or confidence intervals, one the most important
aspect relates to the method conservativeness.  A method that is too
conservative (pessimistic) will tend to provide larger than required intervals
that surpass the required coveraged.  If you are using this to compare systems
(e.g. compare models A and B performance through the same database), then a
too-pessimistic approach may result in overlapping performance intervals for
both systems.  Therefore, it is preferrable to use a technique that is as
precise as possible.

Estimating conservativeness is a difficult task.  The main problem relates to
the underlying hypothesis samples are issued from a binomial distribution
(i.e., are true Bernoulli trials).  Considering that to be true, you can test
the various methods coverage regions through simulation.  The function
:py:func:`bob.measure.curves.estimated_ci_coverage` can help you with that
task, and provides an usage example for the various intervals implemented in
this package:


.. plot::

   import bob.measure

   def betacred_flat_prior(k, l, cov):
       return bob.measure.credible_region.beta(k, l, 1.0, cov)[2:]

   def betacred_jeffreys_prior(k, l, cov):
       return bob.measure.credible_region.beta(k, l, 0.5, cov)[2:]

   samples = 100  # number of samples simulated
   coverage = 0.95

   flat = bob.measure.curves.estimated_ci_coverage(
       betacred_flat_prior, n=samples, expected_coverage=coverage
   )
   bj = bob.measure.curves.estimated_ci_coverage(
       betacred_jeffreys_prior, n=samples, expected_coverage=coverage
   )
   cp = bob.measure.curves.estimated_ci_coverage(
       bob.measure.confidence_interval.clopper_pearson,
       n=samples,
       expected_coverage=coverage,
   )
   ac = bob.measure.curves.estimated_ci_coverage(
       bob.measure.confidence_interval.agresti_coull,
       n=samples,
       expected_coverage=coverage,
   )
   wi = bob.measure.curves.estimated_ci_coverage(
       bob.measure.confidence_interval.wilson,
       n=samples,
       expected_coverage=coverage,
   )

   from matplotlib import pyplot as plt

   plt.plot(wi[0], 100 * wi[1], color="black", label="CI: Wilson (1927)")
   plt.plot(cp[0], 100 * cp[1], color="orange", label="CI: Clopper-Pearson (1934)")
   plt.plot(ac[0], 100 * ac[1], color="purple", label="CI: Agresti-Coull (1998)")
   plt.plot(flat[0], 100 * flat[1], color="blue", label="CR: Beta + Flat Prior (2005)")
   plt.plot(bj[0], 100 * bj[1], color="green", label="CR: Beta + Jeffreys Prior (2005)")

   # Styling
   plt.ylabel(f"Coverage for {100*coverage:.0f}% CR/CI")
   plt.xlabel(f"Success rate (p)")
   plt.title(f"Estimated coverage n={samples}")
   plt.ylim([75, 100])
   plt.hlines(100 * coverage, bj[0][0], bj[0][-1], color="red", linestyle="dashed")
   plt.grid()
   plt.legend()
   plt.show()


Q&A
---

I'm confused, what should I choose?
===================================

You normally want to use the Bayesian approach with the Beta prior, which has a
more natural interpretation.  The frequentist interpretation is harder to
grasp.  The Bayesian approach with a Beta prior offers a good coverage that is
not too conservative.


What if my sampling is not i.i.d.?
==================================

Then using these estimates is not strictly correct, but often done.  If your
samples are not i.i.d. (e.g. phonemes in a speech sequence, pixels in an
image), then these methods will probably provide an overly optimistic estimate
of the interval (i.e., probably the interval for 95% confidence would be larger
if you considered the sample dependence).

.. note::

   Intuition only.  A reference is missing for this.  If you know of anything,
   please patch this description.


How can I calculate the credible interval for an AUC?
=====================================================

This method will also allow you to create plots with confidence intervals for
each ROC curve.

1. Estimate the CI in both directions for each threshold evaluated for the ROC
   curve in question using one of the functions available and the technique
   explored in :py:func:`bob.measure.credible_region.measures`.
2. Assume each threshold's CI forms an ellipse and draw it.
3. Find the inbound and outbound hulls for the ellipses.  These hulls are your
   ROC curve credible intervals.
4. To calculate the AUC credible interval, just calculate the AUC for the
   inbound and outbound curves (hulls).

Currently, this is **not** implemented in this package.


Should I consider the expected value or mode of a credible interval?
====================================================================

TLDR: Use the mode.

As stated in [GOUTTE-2005]_, Section 2.1, if you are using a flat prior
(:math:`\lambda = 1`), then the mode matches the maximum likelihood (ML)
estimate for the indicator in question.  For example, in the case of precision,
the mode of the posterior will be exactly :math:`TP/(TP+FP)`.  The mode is also
the reference for the interval: a confidence interval of 95% is split in half
at each side of the mode.

The expected value will be a smoothed version of the ML estimate
:math:`(TP+1)/(TP+FP+2)` in the case of precision.


Is there a better way to compare 2 systems?
===========================================

TLDR: Use our [GOUTTE-2005]_ implementation to compare two systems using the
F1-score.

In Machine Learning, one typically wants to compare two (or more) systems when
subject to the same input data (samples).  Interval estimates provided in this
package make no assumptions about the underlying samples used in comparing
different systems.  Therefore, intervals provided by simply applying one of the
techniques described before, may be overly pessimistic **in the condition
systems are subject to the same input samples**.

In the specific case two systems are subject to **the same input samples**, a
paired-test may be more adequate.  Unfortunately, for indicators such as
precision, recall or F1-score, many of the paired-tests available in literature
are not adequate [GOUTTE-2005]_.

The main reason for the inadequacy lies on the constraints imposed by such
tests (e.g. gaussianity or symmetry).  In a well-trained system, we would
expect positive and negative samples to be closely located to the extremes of
the system output range (e.g. inside the interval :math:`[0, 1]`).  Besides
being bounded, each of those distributions are likely composed of uni-lateral
tails.  Hence, differences between outputs of systems may not be approximately
normal (more likely bi-modal).  Here is summary of available paired tests and
their assumptions/adequacy:

* Paired (Student's) t-test (parametric): the difference between system outputs
  should be (*approximately*) normally distributed, whereas it is likely
  to be bi-modal.
* Wilcoxon (signed rank) test (non-parametric): does not assume gaussianity on
  the difference of outputs.  It measures how symmetric the difference of
  outputs is around the median.  Assuming symmetry for the difference of two
  heavily tailed distributions would be quite restricting.
* Sign test (non-parametric): Like Wilcoxon test, but without the assumption of
  symmetric distribution of the differences around the median.  This test would
  be adequate for ML system comparison, but is less powerful than the two
  precendent ones.
* Bootstrap: A frequentist approach to credible interval estimation by sampling
  with replacement (differences of) indicators one would want to compare.  This
  would also be adequate as means to compare two systems.


.. include:: links.rst
