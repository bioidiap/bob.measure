#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Thu May 16 11:41:49 CEST 2013

"""Measures for calibration"""

import math
import numpy


def cllr(negatives, positives):
  """Cost of log likelihood ratio as defined by the Bosaris toolkit

  Computes the 'cost of log likelihood ratio' (:math:`C_{llr}`) measure as
  given in the Bosaris toolkit


  Parameters:

    negatives (array): 1D float array that contains the scores of the
      "negative" (noise, non-class) samples of your classifier.

    positives (array): 1D float array that contains the scores of the
      "positive" (signal, class) samples of your classifier.


  Returns:

    float: The computed :math:`C_{llr}` value.

  """
  sum_pos, sum_neg = 0., 0.
  for pos in positives:
    sum_pos += math.log(1. + math.exp(-pos), 2.)
  for neg in negatives:
    sum_neg += math.log(1. + math.exp(neg), 2.)
  return (sum_pos / len(positives) + sum_neg / len(negatives)) / 2.


def min_cllr(negatives, positives):
  """Minimum cost of log likelihood ratio as defined by the Bosaris toolkit

  Computes the 'minimum cost of log likelihood ratio' (:math:`C_{llr}^{min}`)
  measure as given in the bosaris toolkit


  Parameters:

    negatives (array): 1D float array that contains the scores of the
      "negative" (noise, non-class) samples of your classifier.

    positives (array): 1D float array that contains the scores of the
      "positive" (signal, class) samples of your classifier.


  Returns:

    float: The computed :math:`C_{llr}^{min}` value.

  """

  from bob.math import pavx

  # first, sort both scores
  neg = sorted(negatives)
  pos = sorted(positives)
  N = len(neg)
  P = len(pos)
  I = N+P
  # now, iterate through both score sets and add a 0 for negative and 1 for positive scores
  n, p = 0,0
  ideal = numpy.zeros(I)
  neg_indices = [0]*N
  pos_indices = [0]*P
  for i in range(I):
    if p < P and (n == N or neg[n] > pos[p]):
      pos_indices[p] = i
      p += 1
      ideal[i] = 1
    else:
      neg_indices[n] = i
      n += 1

  # compute the pool adjacent violaters method on the ideal LLR scores
  popt = numpy.ndarray(ideal.shape, dtype=numpy.float)
  pavx(ideal, popt)

  # disable runtime warnings for a short time since log(0) will raise a warning
  old_warn_setup = numpy.seterr(divide='ignore')
  # ... compute logs
  posterior_log_odds = numpy.log(popt)-numpy.log(1.-popt);
  log_prior_odds = math.log(float(P)/float(N));
  # ... activate old warnings
  numpy.seterr(**old_warn_setup)

  llrs = posterior_log_odds - log_prior_odds;

  # some weired addition
  #  for i in range(I):
  #    llrs[i] += float(i)*1e-6/float(I)

  # unmix positive and negative scores
  new_neg = numpy.zeros(N)
  for n in range(N):
    new_neg[n] = llrs[neg_indices[n]]
  new_pos = numpy.zeros(P)
  for p in range(P):
    new_pos[p] = llrs[pos_indices[p]]

  # compute cllr of these new 'optimal' LLR scores
  return cllr(new_neg, new_pos)
