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
  """Mean square error between a set of outputs and target values

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
  return numpy.mean((estimation - target)**2, 0)


def rmse (estimation, target):
  """Calculates the root mean square error between a set of outputs and target

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


def relevance (input, machine):
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
  retval = numpy.ndarray((input.shape[1],), 'float64')
  retval.fill(0)
  for k in range(input.shape[1]):
    i2[:,:] = input #reset
    i2[:,k] = numpy.mean(input[:,k])
    retval[k] = (mse(machine(i2), o).sum())**0.5

  return retval


def recognition_rate(cmc_scores, threshold = None, rank = 1):
  """Calculates the recognition rate from the given input

  It is identical to the CMC value for the given ``rank``.

  The input has a specific format, which is a list of two-element tuples.  Each
  of the tuples contains the negative :math:`\\{S_p^-\\}` and the positive
  :math:`\\{S_p^+\\}` scores for one probe item :math:`p`, or ``None`` in case
  of open set recognition.  To read the lists from score files in 4 or 5 column
  format, please use the :py:func:`bob.measure.load.cmc_four_column` or
  :py:func:`bob.measure.load.cmc_five_column` function.

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

    cmc_scores (list): A list in the format ``[(negatives, positives), ...]``
      containing the CMC scores loaded with one of the functions
      (:py:func:`bob.measure.load.cmc_four_column` or
      :py:func:`bob.measure.load.cmc_five_column`).

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
    return 0.

  correct = 0
  counter = 0
  for neg, pos in cmc_scores:
    # set all values that are empty before to None
    if pos is not None and not numpy.array(pos).size:
      pos = None
    if neg is not None and not numpy.array(neg).size:
      neg = None

    if pos is None and neg is None:
      raise ValueError("One pair of the CMC scores has neither positive nor negative values")

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
  item.  To read the lists from score files in 4 or 5 column format, please use
  the :py:func:`bob.measure.load.cmc_four_column` or
  :py:func:`bob.measure.load.cmc_five_column` function.

  For each probe item the probability that the rank :math:`r` of the positive
  score is calculated.  The rank is computed as the number of negative scores
  that are higher than the positive score.  If several positive scores for one
  test item exist, the **highest** positive score is taken. The CMC finally
  computes how many test items have rank r or higher, divided by the total
  number of test values.

  .. note::

     The CMC is not available for open set classification.  Please use the
     :py:func:`detection_identification_rate` and :py:func:`false_alarm_rate`
     instead.


  Parameters:

    cmc_scores (list): A list in the format ``[(negatives, positives), ...]``
      containing the CMC scores loaded with one of the functions
      (:py:func:`bob.measure.load.cmc_four_column` or
      :py:func:`bob.measure.load.cmc_five_column`).

      Each pair contains the ``negative`` and the ``positive`` scores for **one
      probe item**.  Each pair can contain up to one empty array (or ``None``),
      i.e., in case of open set recognition.


  Returns:

    array: A 2D float array representing the CMC curve, with the Rank in the
    first column and the number of correctly classified clients (in this
    rank) in the second column.

  """

  # If no scores are given, we cannot plot anything
  probe_count = float(len(cmc_scores))
  if not probe_count:
    raise ValueError("The given set of scores is empty")

  # compute MC
  match_characteristic = numpy.zeros((max([len(neg) for neg, _ in cmc_scores if neg is not None])+1,), numpy.int)

  for neg, pos in cmc_scores:
    if pos is None or not numpy.array(pos).size:
      raise ValueError("For the CMC computation at least one positive score per pair is necessary.")
    if neg is None:
      neg = []

    # get the maximum positive score for the current probe item
    # (usually, there is only one positive score, but just in case...)
    max_pos = numpy.max(pos)

    # count the number of negative scores that are higher than the best positive score
    index = numpy.sum(neg >= max_pos)
    match_characteristic[index] += 1

  # cumulate
  cumulative_match_characteristic = numpy.cumsum(match_characteristic, dtype=numpy.float64)
  return cumulative_match_characteristic / probe_count


def detection_identification_rate(cmc_scores, threshold, rank = 1):
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

    cmc_scores (list): A list in the format ``[(negatives, positives), ...]``
      containing the CMC scores loaded with one of the functions
      (:py:func:`bob.measure.load.cmc_four_column` or
      :py:func:`bob.measure.load.cmc_five_column`).

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

    index = numpy.sum(neg >= max_pos) # compute the rank (in fact, rank - 1)
    if max_pos >= threshold and index < rank:
      correct += 1

  if not counter:
    logger.warn("No in-gallery probe was found")
    return 0.

  return float(correct) / float(counter)


def false_alarm_rate(cmc_scores, threshold):
  """Computes the `false alarm rate` for the given threshold,.

  This value is designed to be used in an open set identification protocol, and
  defined in Chapter 14.1 of [LiJain2005]_.

  The false alarm rate is designed to be computed on an open set protocol, it
  uses only the probe elements, for which **no** corresponding gallery element
  exists.


  Parameters:

    cmc_scores (list): A list in the format ``[(negatives, positives), ...]``
      containing the CMC scores loaded with one of the functions
      (:py:func:`bob.measure.load.cmc_four_column` or
      :py:func:`bob.measure.load.cmc_five_column`).

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
        raise ValueError("One pair of the CMC scores has neither positive nor negative values")
      if numpy.max(neg) >= threshold:
        incorrect += 1

  if not counter:
    logger.warn("No out-of-gallery probe was found")
    return 0.

  return float(incorrect) / float(counter)


def get_config():
  """Returns a string containing the configuration information.
  """
  import bob.extension
  return bob.extension.get_config(__name__, version.externals)


# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
