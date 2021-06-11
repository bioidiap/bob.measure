#!/usr/bin/env python
# coding=utf-8

"""Methods for global sensitivity analysis"""


from .regression import mse


def relevance(data, f):
    """Calculates the relevance of every input feature to the estimation process

    Uses the formula:

      Neural Triggering System Operating on High Resolution Calorimetry
      Information, Anjos et al, April 2006, Nuclear Instruments and Methods in
      Physics Research, volume 559, pages 134-138

    .. math::

      R(x_{i}) = |E[(o(x) - o(x|x_{i}=E[x_{i}]))^2]|

    In other words, the relevance of a certain input feature **i** is the
    change on the machine output value when such feature is replaced by its
    mean for all input vectors. For this to work, the `input` parameter has to
    be a 2D array with features arranged column-wise while different examples
    are arranged row-wise.


    Parameters
    ==========

    data : numpy.ndarray (float, 2D)

        A 2D float array that contains the input data to be analyzed by the
        method of interest.  Data should be arranged such that the first
        dimension (rows) corresponds to the samples and the second dimension
        (columns) to the features available.  So, if you have 20 samples and 5
        features for each input data sample, then your array should have a
        shape equal to ``(20, 5)``.

    f : object

        A callable, that inputs data, and outputs a :py:class:`numpy.ndarray`.


    Returns
    =======

    relevance : numpy.ndarray (float, 1D)

      A 1D float array as large as the number of columns (second dimension) of
      your input array, estimating the "relevance" of each input column (or
      feature) to the output provided by the function ``f``.

    """

    o = f(data)
    d2 = data.copy()
    retval = numpy.zeros((data.shape[1],), dtype=float)
    for k in range(data.shape[1]):
        d2[:, :] = data  # reset
        d2[:, k] = numpy.mean(data[:, k])
        retval[k] = (mse(f(d2), o).sum()) ** 0.5

    return retval
