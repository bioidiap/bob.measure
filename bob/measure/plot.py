#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Mon 23 May 2011 14:36:14 CEST

import numpy
import warnings
import contextlib


def log_values(min_step=-4, counts_per_step=4):
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

      :py:class:`list`: A list of logarithmically scaled values between
      :math:`10^{M}` and 1.

    """

    import math

    return [
        math.pow(10.0, i * 1.0 / counts_per_step)
        for i in range(min_step * counts_per_step, 0)
    ] + [1.0]


def _semilogx(x, y, **kwargs):
    # remove points were x is 0
    x, y = numpy.asarray(x), numpy.asarray(y)
    zero_index = x == 0
    x = x[~zero_index]
    y = y[~zero_index]
    from matplotlib import pyplot

    return pyplot.semilogx(x, y, **kwargs)


def roc(
    negatives,
    positives,
    npoints=2000,
    CAR=None,
    min_fpr=-8,
    tpr=False,
    semilogx=False,
    **kwargs
):
    """Plots Receiver Operating Characteristic (ROC) curve.

    This method will call ``matplotlib`` to plot the ROC curve for a system which
    contains a particular set of negatives (impostors) and positives (clients)
    scores. We use the standard :py:func:`matplotlib.pyplot.plot` command. All
    parameters passed with exception of the three first parameters of this method
    will be directly passed to the plot command.

    The plot will represent the false-alarm on the horizontal axis and the
    false-rejection on the vertical axis.  The values for the axis will be
    computed using :py:func:`bob.measure.curves.roc`.

    .. note::

      This function does not initiate and save the figure instance, it only
      issues the plotting command. You are the responsible for setting up and
      saving the figure as you see fit.


    Parameters
    ----------
    negatives : array
        1D float array that contains the scores of the
        "negative" (noise, non-class) samples of your classifier. See
        (:py:func:`bob.measure.curves.roc`)

    positives : array
        1D float array that contains the scores of the
        "positive" (signal, class) samples of your classifier. See
        (:py:func:`bob.measure.curves.roc`)

    npoints : :py:class:`int`, optional
        The number of points for the plot. See
        (:py:func:`bob.measure.curves.roc`)

    min_fpr : float, optional
        The minimum value of FPR and FNR that is used for ROC computations.

    tpr : bool, optional
        If True, will plot TPR (TPR = 1 - FNR) on the y-axis instead of FNR.

    semilogx : bool, optional
        If True, will use pyplot.semilogx to plot the ROC curve.

    CAR : :py:class:`bool`, optional
        This option is deprecated. Please use ``TPR`` and ``semilogx`` options instead.
        If set to ``True``, it will plot the CPR
        (CAR) over FPR in using :py:func:`matplotlib.pyplot.semilogx`, otherwise the
        FPR over FNR linearly using :py:func:`matplotlib.pyplot.plot`.

    **kwargs
        Extra plotting parameters, which are
        passed directly to :py:func:`matplotlib.pyplot.plot`.

    Returns
    -------
    object
        `list` of :py:class:`matplotlib.lines.Line2D`: The lines that
        were added as defined by the return value of
        :py:func`matplotlib.pyplot.plot`.
    """
    if CAR is not None:
        warnings.warn(
            "CAR argument is deprecated. Please use TPR and semilogx arguments instead.",
            DeprecationWarning,
        )
        tpr = semilogx = CAR

    from matplotlib import pyplot
    from .curves import roc as calc

    fpr, fnr = calc(negatives, positives, npoints, min_fpr=min_fpr)

    if tpr:
        fnr = 1 - fnr  # plot tpr instead of fnr

    if not semilogx:
        return pyplot.plot(fpr, fnr, **kwargs)
    else:
        return _semilogx(fpr, fnr, **kwargs)


@contextlib.contextmanager
def tight_roc_layout(axes, title=None):
    """Generates a somewhat fancy canvas to draw ROC curves

    Works like a context manager, yielding a figure and an axes set in which
    the ROC curves should be added to.  Once the context is finished,
    ``fig.tight_layout()`` is called.


    Parameters
    ----------

    axes : :py:class:`tuple`, Optional

        Labels for the y and x axes, in this order.

    title : :py:class:`str`, Optional
        Optional title to add to this plot


    Yields
    ------

    figure : matplotlib.figure.Figure
        The figure that should be finally returned to the user

    axes : matplotlib.figure.Axes
        An axis set where to precision-recall plots should be added to

    """

    from matplotlib import pyplot

    fig, axes1 = pyplot.subplots(1)

    # Names and bounds
    axes1.set_ylabel(axes[0])
    axes1.set_xlabel(axes[1])
    axes1.set_xlim([0.0, 1.0])
    axes1.set_ylim([0.0, 1.0])

    if title is not None:
        axes1.set_title(title)

    axes1.grid(linestyle="--", linewidth=1, color="gray", alpha=0.3)

    # we should see some of axes 1 axes
    axes1.spines["right"].set_visible(False)
    axes1.spines["top"].set_visible(False)
    axes1.spines["left"].set_position(("data", -0.015))
    axes1.spines["bottom"].set_position(("data", -0.015))

    # yield execution, lets user draw ROC plots, and the legend before
    # tighteneing the layout
    yield fig, axes1

    pyplot.tight_layout()


def roc_ci(
    negatives,
    positives,
    npoints=2000,
    min_fpr=-8,
    axes=("tpr", "tnr"),
    technique="bayesian|flat",
    coverage=0.95,
    alpha_multiplier=0.3,
    **kwargs
):
    r"""Plots the ROC curve with confidence interval bounds

    This method will call ``matplotlib`` to plot the ROC curve for a system
    which contains a particular set of negatives and positives scores,
    including its credible/confidence interval bounds. We use the standard
    :py:func:`matplotlib.pyplot.plot` command. All parameters passed with
    exception of the three first parameters of this method will be directly
    passed to the plot command.

    .. note::

      This function does not initiate and save the figure instance, it only
      issues the plotting command. You are the responsible for setting up and
      saving the figure as you see fit.


    Parameters
    ----------

    negatives : array
        1D float array that contains the scores of the
        "negative" (noise, non-class) samples of your classifier. See
        (:py:func:`bob.measure.curves.roc`)

    positives : array
        1D float array that contains the scores of the
        "positive" (signal, class) samples of your classifier. See
        (:py:func:`bob.measure.curves.roc`)

    npoints : :py:class:`int`, optional
        The number of points for the plot. See
        (:py:func:`bob.measure.curves.roc`)

    min_fpr : float, optional
        The minimum value of FPR and FNR that is used for ROC computations.

    axes : :py:class:`tuple`, Optional

        Which axes to calculate the curve for.  Variables can be chosen from
        ``tpr``, ``tnr``, ``fnr``, and ``fpr``, ``precision`` (or ``prec``) and
        ``recall`` (or ``rec``, which is an alias for ``tpr``).  Note not all
        combinations make sense (no checks are performed).  You should not try
        to plot, for example, ``tpr`` against ``fnr`` as these rates are
        complementary to 1.0.  The first entry establishes the variable on the
        y axis, the second, on the x axis.

    technique : :py:class:`str`, Optional

        The technique to be used for calculating the confidence/credible
        regions leading to the error bars for each TPR/TNR point.  Available
        implementations are:

        * `bayesian|flat`: uses :py:func:`bob.measure.credible_region.beta`
          with a flat prior (:math:`\lambda=1`)
        * `bayesian|jeffreys`: uses :py:func:`bob.measure.credible_region.beta`
          with Jeffrey's prior (:math:`\lambda=0.5`)
        * `clopper_pearson`: uses
          :py:func:`bob.measure.confidence_interval.clopper_pearson`
        * `agresti_coull`: uses
          :py:func:`bob.measure.confidence_interval.agresti_coull`
        * `wilson`: uses :py:func:`bob.measure.confidence_interval.wilson`

    coverage : :py:class:`float`, Optional
        A floating-point number between 0 and 1.0 indicating the
        coverage you're expecting.  A value of 0.95 will ensure 95%
        of the area under the probability density of the posterior
        is covered by the returned equal-tailed interval.

    alpha_multiplier : :py:class:`float`, Optional
        A value between 0.0 and 1.0 to express the amount of transparence to be
        applied to the confidence/credible margins.  This will be used to
        multiply the ``alpha`` channel of the line itself.  If the default is
        unchanged, then this is the value of the alpha channel on the margins.

    **kwargs
        Extra plotting parameters, which are
        passed directly to :py:func:`matplotlib.pyplot.plot`.

    Returns
    -------

    object
        ``list`` of :py:class:`matplotlib.lines.Line2D`: The lines that
        were added as defined by the return value of
        :py:func`matplotlib.pyplot.plot`.

    auc
        `list` of floats expressing the area under the main curve, lower and
        upper bounds respective.

    """

    from matplotlib import pyplot
    from .curves import roc_ci, curve_ci_hull, area_under_the_curve

    data = roc_ci(
        negatives,
        positives,
        npoints,
        min_fpr=min_fpr,
        axes=axes,
        technique=technique,
        coverage=coverage,
    )

    # This establishes if the curve is reasonably symmetric w.r.t. the origin
    # (y,x) = (0,0), which is the case in homogeneous ROC plots (e.g. TPR vs
    # TNR), and Precision-Recall curves, or w.r.t. (y,x) = (0,1), which is the
    # case when the user wants to plot, e.g., TPR vs FPR.  The value of
    # mixed_rates is False when the curve is symmetric w.r.t. (0,0), and True
    # otherwise.
    mixed_rates = False
    if (axes[0].startswith("t") and axes[1].startswith("f")) or (
        axes[1].startswith("t") and axes[0].startswith("f")
    ):
        mixed_rates = True

    curve, low, high = curve_ci_hull(data, mixed_rates=mixed_rates)
    auc_curve = area_under_the_curve(curve)
    auc_low = area_under_the_curve(low)
    auc_high = area_under_the_curve(high)

    # now we plot middle, lower and upper and paint in the middle
    (line,) = pyplot.plot(curve[1], curve[0], **kwargs)
    color = line.get_color()
    alpha = (
        alpha_multiplier
        if line.get_alpha() is None
        else alpha_multiplier * line.get_alpha()
    )
    (fill,) = pyplot.fill(
        # we concatenate the points so that the formed polygon
        # is structurally coherent (vertices are in the right order)
        numpy.append(high[1], low[1][::-1]),  # x
        numpy.append(high[0], low[0][::-1]),  # y
        # we use the color/alpha from user settings
        color=color,
        alpha=alpha,
        **kwargs,
    )
    return [(line, fill), (auc_curve, auc_low, auc_high)]


def roc_for_far(
    negatives, positives, far_values=log_values(), CAR=True, **kwargs
):
    """Plots the ROC curve for the given list of False Positive Rates (FAR).

    This method will call ``matplotlib`` to plot the ROC curve for a system which
    contains a particular set of negatives (impostors) and positives (clients)
    scores. We use the standard :py:func:`matplotlib.pyplot.semilogx` command.
    All parameters passed with exception of the three first parameters of this
    method will be directly passed to the plot command.

    The plot will represent the False Positive Rate (FPR) on the horizontal
    axis and the Correct Positive Rate (CPR) on the vertical axis.  The values
    for the axis will be computed using :py:func:`bob.measure.curves.roc_for_far`.

    .. note::

      This function does not initiate and save the figure instance, it only
      issues the plotting command. You are the responsible for setting up and
      saving the figure as you see fit.


    Parameters:

      negatives (array): 1D float array that contains the scores of the
        "negative" (noise, non-class) samples of your classifier. See
        (:py:func:`bob.measure.curves.roc`)

      positives (array): 1D float array that contains the scores of the
        "positive" (signal, class) samples of your classifier. See
        (:py:func:`bob.measure.curves.roc`)

      far_values (:py:class:`list`, optional): The values for the FPR, where the
        CPR (CAR) should be plotted; each value should be in range [0,1].

      CAR (:py:class:`bool`, optional): If set to ``True``, it will plot the
        CPR (CAR) over FPR in using :py:func:`matplotlib.pyplot.semilogx`,
        otherwise the FPR over FNR linearly using
        :py:func:`matplotlib.pyplot.plot`.

      kwargs (:py:class:`dict`, optional): Extra plotting parameters, which are
        passed directly to :py:func:`matplotlib.pyplot.plot`.


    Returns:

      :py:class:`list` of :py:class:`matplotlib.lines.Line2D`: The lines that
      were added as defined by the return value of
      :py:func:`matplotlib.pyplot.semilogx`.

    """
    warnings.warn(
        "roc_for_far is deprecated. Please use the roc function instead."
    )

    from matplotlib import pyplot
    from .curves import roc_for_far as calc

    out = calc(negatives, positives, far_values)
    if not CAR:
        return pyplot.plot(out[0, :], out[1, :], **kwargs)
    else:
        return _semilogx(out[0, :], (1 - out[1, :]), **kwargs)


def precision_recall_curve(negatives, positives, npoints=2000, **kwargs):
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
        (:py:func:`bob.measure.curves.precision_recall`)

      positives (array): 1D float array that contains the scores of the
        "positive" (signal, class) samples of your classifier. See
        (:py:func:`bob.measure.curves.precision_recall`)

      npoints (:py:class:`int`, optional): The number of points for the plot. See
        (:py:func:`bob.measure.curves.precision_recall`)

      kwargs (:py:class:`dict`, optional): Extra plotting parameters, which are
        passed directly to :py:func:`matplotlib.pyplot.plot`.


    Returns:

      :py:class:`list` of :py:class:`matplotlib.lines.Line2D`: The lines that
      were added as defined by the return value of
      :py:func:`matplotlib.pyplot.plot`.

    """

    from matplotlib import pyplot
    from .curves import precision_recall as calc

    out = calc(negatives, positives, npoints)
    return pyplot.plot(100.0 * out[0, :], 100.0 * out[1, :], **kwargs)


@contextlib.contextmanager
def tight_pr_layout(title=None):
    """Generates a somewhat fancy canvas to draw Precision-Recall curves

    Works like a context manager, yielding a figure and an axes set in which
    the PR curves should be added to.  Once the context is finished,
    ``fig.tight_layout()`` is called.


    Parameters
    ----------

    title : :py:class:`str`, Optional
        Optional title to add to this plot


    Yields
    ------

    figure : matplotlib.figure.Figure
        The figure that should be finally returned to the user

    axes : matplotlib.figure.Axes
        An axis set where to precision-recall plots should be added to

    """

    from matplotlib import pyplot

    fig, axes1 = pyplot.subplots(1)

    # Names and bounds
    axes1.set_xlabel("Recall")
    axes1.set_ylabel("Precision")
    axes1.set_xlim([0.0, 1.0])
    axes1.set_ylim([0.0, 1.0])

    if title is not None:
        axes1.set_title(title)

    axes1.grid(linestyle="--", linewidth=1, color="gray", alpha=0.2)
    axes2 = axes1.twinx()

    # Annotates plot with F1-score iso-lines
    f_scores = numpy.linspace(0.1, 0.9, num=9)
    tick_locs = []
    tick_labels = []
    for f_score in f_scores:
        x = numpy.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = pyplot.plot(x[y >= 0], y[y >= 0], color="green", alpha=0.1)
        tick_locs.append(y[-1])
        tick_labels.append("%.1f" % f_score)
    axes2.tick_params(axis="y", which="both", pad=0, right=False, left=False)
    axes2.set_ylabel("iso-F", color="green", alpha=0.3)
    axes2.set_ylim([0.0, 1.0])
    axes2.yaxis.set_label_coords(1.015, 0.97)
    axes2.set_yticks(tick_locs)  # notice these are invisible
    for k in axes2.set_yticklabels(tick_labels):
        k.set_color("green")
        k.set_alpha(0.3)
        k.set_size(8)

    # we should see some of axes 1 axes
    axes1.spines["right"].set_visible(False)
    axes1.spines["top"].set_visible(False)
    axes1.spines["left"].set_position(("data", -0.015))
    axes1.spines["bottom"].set_position(("data", -0.015))

    # we shouldn't see any of axes 2 axes
    axes2.spines["right"].set_visible(False)
    axes2.spines["top"].set_visible(False)
    axes2.spines["left"].set_visible(False)
    axes2.spines["bottom"].set_visible(False)

    # yield execution, lets user draw precision-recall plots, and the legend
    # before tighteneing the layout
    yield fig, axes1

    pyplot.tight_layout()


def precision_recall_ci(
    negatives,
    positives,
    npoints=2000,
    min_fpr=-8,
    technique="bayesian|flat",
    coverage=0.95,
    alpha_multiplier=0.3,
    **kwargs
):
    r"""Plots the Precision-Recall curve with confidence interval bounds

    This method will call ``matplotlib`` to plot the Precision-Recall curve for
    a system which contains a particular set of negatives and positives scores,
    including its credible/confidence interval bounds. We use the standard
    :py:func:`matplotlib.pyplot.plot` command. All parameters passed with
    exception of the three first parameters of this method will be directly
    passed to the plot command.

    .. note::

      This function does not initiate and save the figure instance, it only
      issues the plotting command. You are the responsible for setting up and
      saving the figure as you see fit.


    Parameters
    ----------

    negatives : array
        1D float array that contains the scores of the
        "negative" (noise, non-class) samples of your classifier. See
        (:py:func:`bob.measure.curves.roc`)

    positives : array
        1D float array that contains the scores of the
        "positive" (signal, class) samples of your classifier. See
        (:py:func:`bob.measure.curves.roc`)

    npoints : :py:class:`int`, optional
        The number of points for the plot. See
        (:py:func:`bob.measure.curves.roc`)

    min_fpr : float, optional
        The minimum value of FPR and FNR that is used for ROC computations.

    technique : :py:class:`str`, Optional

        The technique to be used for calculating the confidence/credible
        regions leading to the error bars for each TPR/TNR point.  Available
        implementations are:

        * `bayesian|flat`: uses :py:func:`bob.measure.credible_region.beta`
          with a flat prior (:math:`\lambda=1`)
        * `bayesian|jeffreys`: uses :py:func:`bob.measure.credible_region.beta`
          with Jeffrey's prior (:math:`\lambda=0.5`)
        * `clopper_pearson`: uses
          :py:func:`bob.measure.confidence_interval.clopper_pearson`
        * `agresti_coull`: uses
          :py:func:`bob.measure.confidence_interval.agresti_coull`
        * `wilson`: uses :py:func:`bob.measure.confidence_interval.wilson`

    coverage : :py:class:`float`, Optional
        A floating-point number between 0 and 1.0 indicating the
        coverage you're expecting.  A value of 0.95 will ensure 95%
        of the area under the probability density of the posterior
        is covered by the returned equal-tailed interval.

    alpha_multiplier : :py:class:`float`, Optional
        A value between 0.0 and 1.0 to express the amount of transparence to be
        applied to the confidence/credible margins.  This will be used to
        multiply the ``alpha`` channel of the line itself.  If the default is
        unchanged, then this is the value of the alpha channel on the margins.

    **kwargs
        Extra plotting parameters, which are
        passed directly to :py:func:`matplotlib.pyplot.plot`.

    Returns
    -------

    object
        ``list`` of :py:class:`matplotlib.lines.Line2D`: The lines that
        were added as defined by the return value of
        :py:func`matplotlib.pyplot.plot`.

    auc
        `list` of floats expressing the area under the main curve, lower and
        upper bounds respective.

    """

    return roc_ci(
        negatives=negatives,
        positives=positives,
        npoints=npoints,
        min_fpr=min_fpr,
        axes=("precision", "recall"),
        technique=technique,
        coverage=coverage,
        alpha_multiplier=alpha_multiplier,
        **kwargs,
    )


def epc(
    dev_negatives,
    dev_positives,
    test_negatives,
    test_positives,
    npoints=100,
    **kwargs
):
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
        development set. See (:py:func:`bob.measure.curves.epc`)

      dev_positives (array): 1D float array that contains the scores of the
        "positive" (signal, class) samples of your classifier, from the
        development set. See (:py:func:`bob.measure.curves.epc`)

      test_negatives (array): 1D float array that contains the scores of the
        "negative" (noise, non-class) samples of your classifier, from the test
        set. See (:py:func:`bob.measure.curves.epc`)

      test_positives (array): 1D float array that contains the scores of the
        "positive" (signal, class) samples of your classifier, from the test set.
        See (:py:func:`bob.measure.curves.epc`)

      npoints (:py:class:`int`, optional): The number of points for the plot. See
        (:py:func:`bob.measure.curves.epc`)

      kwargs (:py:class:`dict`, optional): Extra plotting parameters, which are
        passed directly to :py:func:`matplotlib.pyplot.plot`.


    Returns:

      :py:class:`list` of :py:class:`matplotlib.lines.Line2D`: The lines that
      were added as defined by the return value of
      :py:func:`matplotlib.pyplot.plot`.

    """

    from matplotlib import pyplot
    from .curves import epc as calc

    out = calc(
        dev_negatives, dev_positives, test_negatives, test_positives, npoints
    )
    return pyplot.plot(out[0, :], 100.0 * out[1, :], **kwargs)


def det(negatives, positives, npoints=2000, min_fpr=-8, **kwargs):
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
      :py:func:`bob.measure.curves.ppndf`.  For example, if you wish to set the x
      and y axis to display data between 1% and 40% here is the recipe:

      .. code-block:: python

        import bob.measure
        from matplotlib import pyplot
        bob.measure.plot.det(...) #call this as many times as you need
        #AFTER you plot the DET curve, just set the axis in this way:
        pyplot.axis([bob.measure.curves.ppndf(k/100.0) for k in (1, 40, 1, 40)])

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
        (:py:func:`bob.measure.curves.det`)

      positives (array): 1D float array that contains the scores of the
        "positive" (signal, class) samples of your classifier. See
        (:py:func:`bob.measure.curves.det`)

      npoints (:py:class:`int`, optional): The number of points for the plot. See
        (:py:func:`bob.measure.curves.det`)

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
        "0.000001",
        "0.000002",
        "0.000005",
        "0.00001",
        "0.00002",
        "0.00005",
        "0.0001",
        "0.0002",
        "0.0005",
        "0.001",
        "0.002",
        "0.005",
        "0.01",
        "0.02",
        "0.05",
        "0.1",
        "0.2",
        "0.4",
        "0.6",
        "0.8",
        "0.9",
        "0.95",
        "0.98",
        "0.99",
        "0.995",
        "0.998",
        "0.999",
        "0.9995",
        "0.9998",
        "0.9999",
        "0.99995",
        "0.99998",
        "0.99999",
    ]

    desiredLabels = [
        "0.0001",
        "0.0002",
        "0.0005",
        "0.001",
        "0.002",
        "0.005",
        "0.01",
        "0.02",
        "0.05",
        "0.1",
        "0.2",
        "0.5",
        "1",
        "2",
        "5",
        "10",
        "20",
        "40",
        "60",
        "80",
        "90",
        "95",
        "98",
        "99",
        "99.5",
        "99.8",
        "99.9",
        "99.95",
        "99.98",
        "99.99",
        "99.995",
        "99.998",
        "99.999",
    ]

    # this will actually do the plotting
    from matplotlib import pyplot
    from .curves import det as calc
    from .curves import ppndf

    out = calc(negatives, positives, npoints, min_fpr)
    retval = pyplot.plot(out[0, :], out[1, :], **kwargs)

    # now the trick: we must plot the tick marks by hand using the PPNDF method
    pticks = ppndf(numpy.array(desiredTicks, dtype=float))
    ax = pyplot.gca()  # and finally we set our own tick marks
    ax.set_xticks(pticks)
    ax.set_xticklabels(desiredLabels)
    ax.set_yticks(pticks)
    ax.set_yticklabels(desiredLabels)

    return retval


def det_axis(v, **kwargs):
    """Sets the axis in a DET plot.

    This method wraps the :py:func:`matplotlib.pyplot.axis` by calling
    :py:func:`bob.measure.curves.ppndf` on the values passed by the user so they
    are meaningful in a DET plot as performed by :py:func:`bob.measure.plot.det`.


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
    from .curves import ppndf

    # treat input
    try:
        tv = list(v)  # normal input
        if len(tv) != 4:
            raise IndexError
        tv = [ppndf(float(k) / 100) for k in tv]
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


def cmc(cmc_scores, logx=True, **kwargs):
    """Plots the (cumulative) match characteristics and returns the maximum rank.

    This function plots a CMC curve using the given CMC scores (:py:class:`list`:
        A list of tuples, where each tuple contains the
        ``negative`` and ``positive`` scores for one probe of the database).


    Parameters:

      cmc_scores (array): 1D float array containing the CMC values (See
        :py:func:`bob.measure.cmc.cmc`)

      logx (:py:class:`bool`, optional): If set (the default), plots the rank
        axis in logarithmic scale using :py:func:`matplotlib.pyplot.semilogx` or
        in linear scale using :py:func:`matplotlib.pyplot.plot`

      kwargs (:py:class:`dict`, optional): Extra plotting parameters, which are
        passed directly to :py:func:`matplotlib.pyplot.plot`.


    Returns:

      int: The number of classes (clients) in the given scores.

    """

    from matplotlib import pyplot
    from .cmc import cmc as calc

    out = calc(cmc_scores)

    if logx:
        _semilogx(range(1, len(out) + 1), out, **kwargs)
    else:
        pyplot.plot(range(1, len(out) + 1), out, **kwargs)

    return len(out)


def detection_identification_curve(
    cmc_scores, far_values=log_values(), rank=1, logx=True, **kwargs
):
    """Plots the Detection & Identification curve over the FPR

    This curve is designed to be used in an open set identification protocol, and
    defined in Chapter 14.1 of [LiJain2005]_.  It requires to have at least one
    open set probe item, i.e., with no corresponding gallery, such that the
    positives for that pair are ``None``.

    The detection and identification curve first computes FPR thresholds based on
    the out-of-set probe scores (negative scores).  For each probe item, the
    **maximum** negative score is used.  Then, it plots the detection and
    identification rates for those thresholds, which are based on the in-set
    probe scores only. See [LiJain2005]_ for more details.

    .. [LiJain2005] **Stan Li and Anil K. Jain**, *Handbook of Face Recognition*, Springer, 2005


    Parameters:

      cmc_scores (array): 1D float array containing the CMC values (See
        :py:func:`bob.measure.cmc.cmc`)

      rank (:py:class:`int`, optional): The rank for which the curve should be
        plotted

      far_values (:py:class:`list`, optional): The values for the FPR (FAR), where the
        CPR (CAR) should be plotted; each value should be in range [0,1].

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
    import math
    from matplotlib import pyplot
    from .brute_force import far_threshold
    from .cmc import detection_identification_rate

    # for each probe, for which no positives exists, get the highest negative
    # score; and sort them to compute the FAR thresholds
    negatives = sorted(
        max(neg)
        for neg, pos in cmc_scores
        if (pos is None or not numpy.array(pos).size) and neg is not None
    )
    if not negatives:
        raise ValueError(
            "There need to be at least one pair with only negative scores"
        )

    # compute thresholds based on FAR values
    thresholds = [far_threshold(negatives, [], v, True) for v in far_values]

    # compute detection and identification rate based on the thresholds for
    # the given rank
    rates = [
        detection_identification_rate(cmc_scores, t, rank)
        if not math.isnan(t)
        else numpy.nan
        for t in thresholds
    ]

    # plot curve
    if logx:
        return _semilogx(far_values, rates, **kwargs)
    else:
        return pyplot.plot(far_values, rates, **kwargs)
