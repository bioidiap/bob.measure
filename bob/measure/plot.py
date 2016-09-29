#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Mon 23 May 2011 14:36:14 CEST


def log_values(min_step = -4, counts_per_step = 4):
  """Computes log-scaled values between :math:`10^{M}` and 1

  This function computes log-scaled values between :math:`10^{M}` and 1
  (including), where :math:`M` is the ``min_ste`` argument, which needs to be a
  negative integer.  The integral ``counts_per_step`` value defines how many
  values between two adjacent powers of 10 will be created.  The total number
  of values will be ``-min_step * counts_per_step + 1``.


  Parameters:

    min_step (:py:class:`int`, optional): The power of 10 that will be the
      minimum value.  E.g., the default ``-4`` will result in the first number
      to be :math:`10^{-4}` = ``0.00001`` or ``0.01%``

    counts_per_step (:py:class:`int`, optional): The number of values that will
      be put between two adjacent powers of 10.  With the default value ``4``
      (and default values of ``min_step``), we will get ``log_list[0] ==
      1e-4``, ``log_list[4] == 1e-3``, ..., ``log_list[16] == 1``.


  Returns:

    list: A list of logarithmically scaled values between :math:`10^{M}` and 1.

  """

  import math
  return [math.pow(10., i * 1./counts_per_step) for i in range(min_step*counts_per_step,0)] + [1.]


def roc(negatives, positives, npoints=100, CAR=False, **kwargs):
  """Plots Receiver Operating Characteristic (ROC) curve.

  This method will call ``matplotlib`` to plot the ROC curve for a system which
  contains a particular set of negatives (impostors) and positives (clients)
  scores. We use the standard :py:func:`matplotlib.pyplot.plot` command. All
  parameters passed with exception of the three first parameters of this method
  will be directly passed to the plot command.

  The plot will represent the false-alarm on the horizontal axis and the
  false-rejection on the vertical axis.  The values for the axis will be
  computed using :py:func:`bob.measure.roc`.

  .. note::

    This function does not initiate and save the figure instance, it only
    issues the plotting command. You are the responsible for setting up and
    saving the figure as you see fit.


  Parameters:

    negatives (array): 1D float array that contains the scores of the
      "negative" (noise, non-class) samples of your classifier. See
      (:py:func:`bob.measure.roc`)

    positives (array): 1D float array that contains the scores of the
      "positive" (signal, class) samples of your classifier. See
      (:py:func:`bob.measure.roc`)

    npoints (:py:class:`int`, optional): The number of points for the plot. See
      (:py:func:`bob.measure.roc`)

    CAR (:py:class:`bool`, optional): If set to ``True``, it will plot the CAR
      over FAR in using :py:func:`matplotlib.pyplot.semilogx`, otherwise the
      FAR over FRR linearly using :py:func:`matplotlib.pyplot.plot`.

    kwargs (:py:class:`dict`, optional): Extra plotting parameters, which are
      passed directly to :py:func:`matplotlib.pyplot.plot`.


  Returns:

    :py:class:`list` of :py:class:`matplotlib.lines.Line2D`: The lines that
    were added as defined by the return value of
    :py:func:`matplotlib.pyplot.plot`.

  """

  from matplotlib import pyplot
  from . import roc as calc
  out = calc(negatives, positives, npoints)
  if not CAR:
    return pyplot.plot(100.0*out[0,:], 100.0*out[1,:], **kwargs)
  else:
    return pyplot.semilogx(100.0*out[0,:], 100.0*(1-out[1,:]), **kwargs)


def roc_for_far(negatives, positives, far_values = log_values(), **kwargs):
  """Plots the ROC curve for the given list of False Acceptance Rates (FAR).

  This method will call ``matplotlib`` to plot the ROC curve for a system which
  contains a particular set of negatives (impostors) and positives (clients)
  scores. We use the standard :py:func:`matplotlib.pyplot.semilogx` command.
  All parameters passed with exception of the three first parameters of this
  method will be directly passed to the plot command.

  The plot will represent the False Acceptance Rate (FAR) on the horizontal
  axis and the Correct Acceptance Rate (CAR) on the vertical axis.  The values
  for the axis will be computed using :py:func:`bob.measure.roc_for_far`.

  .. note::

    This function does not initiate and save the figure instance, it only
    issues the plotting command. You are the responsible for setting up and
    saving the figure as you see fit.


  Parameters:

    negatives (array): 1D float array that contains the scores of the
      "negative" (noise, non-class) samples of your classifier. See
      (:py:func:`bob.measure.roc`)

    positives (array): 1D float array that contains the scores of the
      "positive" (signal, class) samples of your classifier. See
      (:py:func:`bob.measure.roc`)

    far_values (:py:class:`list`, optional): The values for the FAR, where the
      CAR should be plotted; each value should be in range [0,1].

    kwargs (:py:class:`dict`, optional): Extra plotting parameters, which are
      passed directly to :py:func:`matplotlib.pyplot.plot`.


  Returns:

    :py:class:`list` of :py:class:`matplotlib.lines.Line2D`: The lines that
    were added as defined by the return value of
    :py:func:`matplotlib.pyplot.semilogx`.

  """

  from matplotlib import pyplot
  from . import roc_for_far as calc
  out = calc(negatives, positives, far_values)
  return pyplot.semilogx(100.0*out[0,:], 100.0*(1-out[1,:]), **kwargs)


def precision_recall_curve(negatives, positives, npoints=100, **kwargs):
  """Plots a Precision-Recall curve.

  This method will call ``matplotlib`` to plot the precision-recall curve for a
  system which contains a particular set of ``negatives`` (impostors) and
  ``positives`` (clients) scores. We use the standard
  :py:func:`matplotlib.pyplot.plot` command. All parameters passed with
  exception of the three first parameters of this method will be directly
  passed to the plot command.

  .. note::

    This function does not initiate and save the figure instance, it only
    issues the plotting command. You are the responsible for setting up and
    saving the figure as you see fit.


  Parameters:

    negatives (array): 1D float array that contains the scores of the
      "negative" (noise, non-class) samples of your classifier. See
      (:py:func:`bob.measure.precision_recall_curve`)

    positives (array): 1D float array that contains the scores of the
      "positive" (signal, class) samples of your classifier. See
      (:py:func:`bob.measure.precision_recall_curve`)

    npoints (:py:class:`int`, optional): The number of points for the plot. See
      (:py:func:`bob.measure.precision_recall_curve`)

    kwargs (:py:class:`dict`, optional): Extra plotting parameters, which are
      passed directly to :py:func:`matplotlib.pyplot.plot`.


  Returns:

    :py:class:`list` of :py:class:`matplotlib.lines.Line2D`: The lines that
    were added as defined by the return value of
    :py:func:`matplotlib.pyplot.plot`.

  """

  from matplotlib import pyplot
  from . import precision_recall_curve as calc
  out = calc(negatives, positives, npoints)
  return pyplot.plot(100.0*out[0,:], 100.0*out[1,:], **kwargs)


def epc(dev_negatives, dev_positives, test_negatives, test_positives,
    npoints=100, **kwargs):
  """Plots Expected Performance Curve (EPC) as defined in the paper:

  Bengio, S., Keller, M., Mariéthoz, J. (2004). The Expected Performance Curve.
  International Conference on Machine Learning ICML Workshop on ROC Analysis in
  Machine Learning, 136(1), 1963–1966. IDIAP RR. Available:
  http://eprints.pascal-network.org/archive/00000670/

  This method will call ``matplotlib`` to plot the EPC curve for a system which
  contains a particular set of negatives (impostors) and positives (clients)
  for both the development and test sets. We use the standard
  :py:func:`matplotlib.pyplot.plot` command. All parameters passed with
  exception of the five first parameters of this method will be directly passed
  to the plot command.

  The plot will represent the minimum HTER on the vertical axis and the cost on
  the horizontal axis.

  .. note::

    This function does not initiate and save the figure instance, it only
    issues the plotting commands. You are the responsible for setting up and
    saving the figure as you see fit.


  Parameters:

    dev_negatives (array): 1D float array that contains the scores of the
      "negative" (noise, non-class) samples of your classifier, from the
      development set. See (:py:func:`bob.measure.epc`)

    dev_positives (array): 1D float array that contains the scores of the
      "positive" (signal, class) samples of your classifier, from the
      development set. See (:py:func:`bob.measure.epc`)

    test_negatives (array): 1D float array that contains the scores of the
      "negative" (noise, non-class) samples of your classifier, from the test
      set. See (:py:func:`bob.measure.epc`)

    test_positives (array): 1D float array that contains the scores of the
      "positive" (signal, class) samples of your classifier, from the test set.
      See (:py:func:`bob.measure.epc`)

    npoints (:py:class:`int`, optional): The number of points for the plot. See
      (:py:func:`bob.measure.epc`)

    kwargs (:py:class:`dict`, optional): Extra plotting parameters, which are
      passed directly to :py:func:`matplotlib.pyplot.plot`.


  Returns:

    :py:class:`list` of :py:class:`matplotlib.lines.Line2D`: The lines that
    were added as defined by the return value of
    :py:func:`matplotlib.pyplot.plot`.

  """

  from matplotlib import pyplot
  from . import epc as calc

  out = calc(dev_negatives, dev_positives, test_negatives, test_positives,
      npoints)
  return pyplot.plot(out[0,:], 100.0*out[1,:], **kwargs)


def det(negatives, positives, npoints=100, axisfontsize='x-small', **kwargs):
  """Plots Detection Error Trade-off (DET) curve as defined in the paper:

  Martin, A., Doddington, G., Kamm, T., Ordowski, M., & Przybocki, M. (1997).
  The DET curve in assessment of detection task performance. Fifth European
  Conference on Speech Communication and Technology (pp. 1895-1898). Available:
  http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.117.4489&rep=rep1&type=pdf

  This method will call ``matplotlib`` to plot the DET curve(s) for a system
  which contains a particular set of negatives (impostors) and positives
  (clients) scores. We use the standard :py:func:`matplotlib.pyplot.plot`
  command. All parameters passed with exception of the three first parameters
  of this method will be directly passed to the plot command.

  The plot will represent the false-alarm on the horizontal axis and the
  false-rejection on the vertical axis.

  This method is strongly inspired by the NIST implementation for Matlab,
  called DETware, version 2.1 and available for download at the NIST website:

  http://www.itl.nist.gov/iad/mig/tools/

  .. note::

    This function does not initiate and save the figure instance, it only
    issues the plotting commands. You are the responsible for setting up and
    saving the figure as you see fit.

  .. note::

    If you wish to reset axis zooming, you must use the Gaussian scale rather
    than the visual marks showed at the plot, which are just there for
    displaying purposes. The real axis scale is based on
    :py:func:`bob.measure.ppndf`.  For example, if you wish to set the x and y
    axis to display data between 1% and 40% here is the recipe:

    .. code-block:: python

      import bob.measure
      from matplotlib import pyplot
      bob.measure.plot.det(...) #call this as many times as you need
      #AFTER you plot the DET curve, just set the axis in this way:
      pyplot.axis([bob.measure.ppndf(k/100.0) for k in (1, 40, 1, 40)])

    We provide a convenient way for you to do the above in this module. So,
    optionally, you may use the :py:func:`bob.measure.plot.det_axis` method
    like this:

    .. code-block:: python

      import bob.measure
      bob.measure.plot.det(...)
      # please note we convert percentage values in det_axis()
      bob.measure.plot.det_axis([1, 40, 1, 40])


  Parameters:

    negatives (array): 1D float array that contains the scores of the
      "negative" (noise, non-class) samples of your classifier. See
      (:py:func:`bob.measure.det`)

    positives (array): 1D float array that contains the scores of the
      "positive" (signal, class) samples of your classifier. See
      (:py:func:`bob.measure.det`)

    npoints (:py:class:`int`, optional): The number of points for the plot. See
      (:py:func:`bob.measure.det`)

    axisfontsize (:py:class:`str`, optional): The size to be used by
      x/y-tick-labels to set the font size on the axis

    kwargs (:py:class:`dict`, optional): Extra plotting parameters, which are
      passed directly to :py:func:`matplotlib.pyplot.plot`.


  Returns:

    :py:class:`list` of :py:class:`matplotlib.lines.Line2D`: The lines that
    were added as defined by the return value of
    :py:func:`matplotlib.pyplot.plot`.

  """

  # these are some constants required in this method
  desiredTicks = [
      "0.00001", "0.00002", "0.00005",
      "0.0001", "0.0002", "0.0005",
      "0.001", "0.002", "0.005",
      "0.01", "0.02", "0.05",
      "0.1", "0.2", "0.4", "0.6", "0.8", "0.9",
      "0.95", "0.98", "0.99",
      "0.995", "0.998", "0.999",
      "0.9995", "0.9998", "0.9999",
      "0.99995", "0.99998", "0.99999"
      ]

  desiredLabels = [
      "0.001", "0.002", "0.005",
      "0.01", "0.02", "0.05",
      "0.1", "0.2", "0.5",
      "1", "2", "5",
      "10", "20", "40", "60", "80", "90",
      "95", "98", "99",
      "99.5", "99.8", "99.9",
      "99.95", "99.98", "99.99",
      "99.995", "99.998", "99.999"
      ]

  # this will actually do the plotting
  from matplotlib import pyplot
  from . import det as calc
  from . import ppndf

  out = calc(negatives, positives, npoints)
  retval = pyplot.plot(out[0,:], out[1,:], **kwargs)

  # now the trick: we must plot the tick marks by hand using the PPNDF method
  pticks = [ppndf(float(v)) for v in desiredTicks]
  ax = pyplot.gca() #and finally we set our own tick marks
  ax.set_xticks(pticks)
  ax.set_xticklabels(desiredLabels, size=axisfontsize)
  ax.set_yticks(pticks)
  ax.set_yticklabels(desiredLabels, size=axisfontsize)

  return retval


def det_axis(v, **kwargs):
  """Sets the axis in a DET plot.

  This method wraps the :py:func:`matplotlib.pyplot.axis` by calling
  :py:func:`bob.measure.ppndf` on the values passed by the user so they are
  meaningful in a DET plot as performed by :py:func:`bob.measure.plot.det`.


  Parameters:

    v (``sequence``): A sequence (list, tuple, array or the like) containing
      the X and Y limits in the order ``(xmin, xmax, ymin, ymax)``. Expected
      values should be in percentage (between 0 and 100%).  If ``v`` is not a
      list or tuple that contains 4 numbers it is passed without further
      inspection to :py:func:`matplotlib.pyplot.axis`.

    kwargs (:py:class:`dict`, optional): Extra plotting parameters, which are
      passed directly to :py:func:`matplotlib.pyplot.axis`.


  Returns:

    object: Whatever is returned by :py:func:`matplotlib.pyplot.axis`.

  """

  import logging
  logger = logging.getLogger("bob.measure")

  from matplotlib import pyplot
  from . import ppndf

  # treat input
  try:
    tv = list(v) #normal input
    if len(tv) != 4: raise IndexError
    tv = [ppndf(float(k)/100) for k in tv]
    cur = pyplot.axis()

    # limits must be within bounds
    if tv[0] < cur[0]:
      logger.warn("Readjusting xmin: the provided value is out of bounds")
      tv[0] = cur[0]
    if tv[1] > cur[1]:
      logger.warn("Readjusting xmax: the provided value is out of bounds")
      tv[1] = cur[1]
    if tv[2] < cur[2]:
      logger.warn("Readjusting ymin: the provided value is out of bounds")
      tv[2] = cur[2]
    if tv[3] > cur[3]:
      logger.warn("Readjusting ymax: the provided value is out of bounds")
      tv[3] = cur[3]

  except:
    tv = v

  return pyplot.axis(tv, **kwargs)


def cmc(cmc_scores, logx = True, **kwargs):
  """Plots the (cumulative) match characteristics and returns the maximum rank.

  This function plots a CMC curve using the given CMC scores, which can be read
  from the our score files using the
  :py:func:`bob.measure.load.cmc_four_column` or
  :py:func:`bob.measure.load.cmc_five_column` methods.  The structure of the
  ``cmc_scores`` parameter is relatively complex.  It contains a list of pairs
  of lists.  For each probe object, a pair of list negative and positive scores
  is required.


  Parameters:

    cmc_scores (array): 1D float array containing the CMC values (See
      :py:func:`bob.measure.cmc`)

    logx (:py:class:`bool`, optional): If set (the default), plots the rank
      axis in logarithmic scale using :py:func:`matplotlib.pyplot.semilogx` or
      in linear scale using :py:func:`matplotlib.pyplot.plot`

    kwargs (:py:class:`dict`, optional): Extra plotting parameters, which are
      passed directly to :py:func:`matplotlib.pyplot.plot`.


  Returns:

    int: The number of classes (clients) in the given scores.

  """

  from matplotlib import pyplot
  from . import cmc as calc

  out = calc(cmc_scores)

  if logx:
    pyplot.semilogx(range(1, len(out)+1), out * 100, **kwargs)
  else:
    pyplot.plot(range(1, len(out)+1), out * 100, **kwargs)

  return len(out)


def detection_identification_curve(cmc_scores, far_values = log_values(), rank
    = 1, logx = True, **kwargs):
  """Plots the Detection & Identification curve over the FAR

  This curve is designed to be used in an open set identification protocol, and
  defined in Chapter 14.1 of [LiJain2005]_.  It requires to have at least one
  open set probe item, i.e., with no corresponding gallery, such that the
  positives for that pair are ``None``.

  The detection and identification curve first computes FAR thresholds based on
  the out-of-set probe scores (negative scores).  For each probe item, the
  **maximum** negative score is used.  Then, it plots the detection and
  identification rates for those thresholds, which are based on the in-set
  probe scores only. See [LiJain2005]_ for more details.

  .. [LiJain2005] **Stan Li and Anil K. Jain**, *Handbook of Face Recognition*, Springer, 2005


  Parameters:

    cmc_scores (array): 1D float array containing the CMC values (See
      :py:func:`bob.measure.cmc`)

    rank (:py:class:`int`, optional): The rank for which the curve should be
      plotted

    far_values (:py:class:`list`, optional): The values for the FAR, where the
      CAR should be plotted; each value should be in range [0,1].

    logx (:py:class:`bool`, optional): If set (the default), plots the rank
      axis in logarithmic scale using :py:func:`matplotlib.pyplot.semilogx` or
      in linear scale using :py:func:`matplotlib.pyplot.plot`

    kwargs (:py:class:`dict`, optional): Extra plotting parameters, which are
      passed directly to :py:func:`matplotlib.pyplot.plot`.


  Returns:

    :py:class:`list` of :py:class:`matplotlib.lines.Line2D`: The lines that
    were added as defined by the return value of
    :py:func:`matplotlib.pyplot.plot`.

  """

  import numpy
  from matplotlib import pyplot
  from . import far_threshold, detection_identification_rate

  # for each probe, for which no positives exists, get the highest negative score; and sort them to compute the FAR thresholds
  negatives = sorted(max(neg) for neg,pos in cmc_scores if (pos is None or not numpy.array(pos).size) and neg is not None)
  if not negatives:
    raise ValueError("There need to be at least one pair with only negative scores")

  # compute thresholds based on FAR values
  thresholds = [far_threshold(negatives, [], v, True) for v in far_values]

  # compute detection and identification rate based on the thresholds for the given rank
  rates = [100.*detection_identification_rate(cmc_scores, t, rank) for t in thresholds]

  # plot curve
  if logx:
    return pyplot.semilogx(far_values, rates, **kwargs)
  else:
    return pyplot.plot(far_values, rates, **kwargs)
