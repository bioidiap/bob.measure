.. coding=utf-8

.. testsetup:: *

   from matplotlib import pyplot as plt
   import bob.measure


Credible Regions and Confidence Intervals
=========================================

A confidence interval for parameter :math:`x` consists of a lower estimate
:math:`L`, and an upper estimate :math:`U`, such that the probability of the
true value being within the interval estimate is equal to :math:`\alpha`.  For
example, a 95% confidence interval (i.e. :math:`\alpha = 0.95`) for a parameter
:math:`x` is given by :math:`[L, U]` such that

.. math:: Prob(x ∈ [L,U]) = 95%

The smaller the test size, the wider the confidence interval will be, and the
greater :math:`\alpha`, the smaller the confidence interval will be.

[CLOPPER-1934]_ develops a common method for calculating confidence intervals,
as function of the number of success, the number of trials and confidence value
:math:`\alpha` is used as :py:func:`bob.measure.confidence.clopper_pearson`.
It is based on the cumulative probabilities of the binomial distribution. This
method is quite conservative, meaning that the true coverage rate of a 95%
Clopper–Pearson interval may be well above 95%.

For example, we want to evaluate the reliability of a system to identify
registered persons.  Let's say that among 10,000 accepted transactions, 9856 are
true matches. The 95% confidence interval for true match rate is then:

.. doctest:: python

   >>> numpy.allclose(bob.measure.confidence_interval.clopper_pearson(9856, 144),(0.98306835053282549, 0.98784270928084694))
   True

meaning there is a 95% probability that the true match rate is inside
:math:`[0.983, 0.988]`.
