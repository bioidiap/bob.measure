# import Libraries of other lib packages
import bob.math
import bob.io.base

from ._library import *
from . import version
from .version import module as __version__

from . import plot
from . import load
from . import calibration
from . import openbr
import numpy

def mse (estimation, target):
  """mse(estimation, target) -> error

  Calculates the mean square error between a set of outputs and target
  values using the following formula:

  .. math::

    MSE(\hat{\Theta}) = E[(\hat{\Theta} - \Theta)^2]

  Estimation (:math:`\hat{\Theta}`) and target (:math:`\Theta`) are supposed to
  have 2 dimensions. Different examples are organized as rows while different
  features in the estimated values or targets are organized as different
  columns.
  """
  return numpy.mean((estimation - target)**2, 0)

def rmse (estimation, target):
  """rmse(estimation, target) -> error

  Calculates the root mean square error between a set of outputs and target
  values using the following formula:

  .. math::

    RMSE(\hat{\Theta}) = \sqrt(E[(\hat{\Theta} - \Theta)^2])

  Estimation (:math:`\hat{\Theta}`) and target (:math:`\Theta`) are supposed to
  have 2 dimensions. Different examples are organized as rows while different
  features in the estimated values or targets are organized as different
  columns.
  """
  return numpy.sqrt(mse(estimation, target))

def relevance (input, machine):
  """relevance (input, machine) -> relevances

  Calculates the relevance of every input feature to the estimation process
  using the following definition from:

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
  """

  o = machine(input)
  i2 = input.copy()
  retval = numpy.ndarray((input.shape[1],), 'float64')
  retval.fill(0)
  for k in range(input.shape[1]):
    i2[:,:] = input #reset
    i2[:,k] = numpy.mean(input[:,k])
    retval[k] = (mse(machine(i2), o).sum())**0.5

  return retval

def recognition_rate(cmc_scores):
  """recognition_rate(cmc_scores) -> RR

  Calculates the recognition rate from the given input, which is identical
  to the rank 1 (C)MC value.

  The input has a specific format, which is a list of two-element tuples. Each
  of the tuples contains the negative and the positive scores for one test
  item.  To read the lists from score files in 4 or 5 column format, please use
  the :py:func:`bob.measure.load.cmc_four_column` or
  :py:func:`bob.measure.load.cmc_five_column` function.

  The recognition rate is defined as the number of test items, for which the
  positive score is greater than or equal to all negative scores, divided by
  the number of all test items.  If several positive scores for one test item
  exist, the **highest** score is taken.

  **Parameters:**

  ``cmc_scores`` : [(array_like(1D, float), array_like(1D, float))]
    A list of tuples, where each tuple contains the ``negative`` and ``positive`` scores for one probe of the database

  **Returns:**

  ``RR`` : float
    The rank 1 recognition rate, i.e., the relative number of correctly identified identities
  """
  # If no scores are given, the recognition rate is exactly 0.
  if not cmc_scores:
    return 0.

  correct = 0.
  for neg, pos in cmc_scores:
    # get the maximum positive score for the current probe item
    # (usually, there is only one positive score, but just in case...)
    max_pos = numpy.max(pos)
    # check if the positive score is smaller than all negative scores
    if (neg < max_pos).all():
      correct += 1.

  # return relative number of correctly matched scores
  return correct / float(len(cmc_scores))

def cmc(cmc_scores):
  """cmc(cmc_scores) -> curve

  Calculates the cumulative match characteristic (CMC) from the given input.

  The input has a specific format, which is a list of two-element tuples. Each
  of the tuples contains the negative and the positive scores for one test
  item.  To read the lists from score files in 4 or 5 column format, please use
  the :py:func:`bob.measure.load.cmc_four_column` or
  :py:func:`bob.measure.load.cmc_five_column` function.

  For each test item the probability that the rank r of the positive score is
  calculated.  The rank is computed as the number of negative scores that are
  higher than the positive score.  If several positive scores for one test item
  exist, the **highest** positive score is taken. The CMC finally computes how
  many test items have rank r or higher.

  **Parameters:**

  ``cmc_scores`` : [(array_like(1D, float), array_like(1D, float))]
    A list of tuples, where each tuple contains the ``negative`` and ``positive`` scores for one probe of the database

  **Returns:**

  ``curve`` : array_like(2D, float)
    The CMC curve, with the Rank in the first column and the number of correctly classified clients (in this rank) in the second column.
  """

  # If no scores are given, we cannot plot anything
  probe_count = float(len(cmc_scores))
  if not probe_count:
    raise ValueError("The given set of scores is empty")

  # compute MC
  match_characteristic = numpy.zeros((max([len(neg) for (neg,pos) in cmc_scores])+1,), numpy.int)
  for neg, pos in cmc_scores:
    # get the maximum positive score for the current probe item
    # (usually, there is only one positive score, but just in case...)
    max_pos = numpy.max(pos)
    # count the number of negative scores that are higher than the best positive score
    index = numpy.sum(neg >= max_pos)
    match_characteristic[index] += 1

  # cumulate
  cumulative_match_characteristic = numpy.ndarray(match_characteristic.shape, numpy.float64)
  count = 0.
  for i in range(match_characteristic.shape[0]):
    count += match_characteristic[i]
    cumulative_match_characteristic[i] = count / probe_count

  return cumulative_match_characteristic


def get_config():
  """Returns a string containing the configuration information.
  """
  import bob.extension
  return bob.extension.get_config(__name__, version.externals)


# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
