#!/usr/bin/env python
# coding=utf-8


import numpy


def mse(estimation, target):
    """Mean square error between a set of outputs and target values

    Uses the formula:

    .. math::

      MSE(\hat{\Theta}) = E[(\hat{\Theta} - \Theta)^2]

    Estimation (:math:`\hat{\Theta}`) and target (:math:`\Theta`) are supposed
    to have 2 dimensions. Different examples are organized as rows while
    different features in the estimated values or targets are organized as
    different columns.


    Parameters
    ==========

    estimation : numpy.ndarray (float)

      A N-dimensional array that corresponds to the value estimated by your
      procedure

    target : numpy.ndarray (float)

      A N-dimensional array that corresponds to the expected value


    Returns
    =======

    mse : float

      The average of the squared error between the estimated value and the
      target

    """
    return numpy.mean((estimation - target) ** 2, 0)


def rmse(estimation, target):
    """Calculates the root mean square error between a set of outputs and target

    Uses the formula:

    .. math::

       RMSE(\hat{\Theta}) = \sqrt(E[(\hat{\Theta} - \Theta)^2])

    Estimation (:math:`\hat{\Theta}`) and target (:math:`\Theta`) are supposed
    to have 2 dimensions. Different examples are organized as rows while
    different features in the estimated values or targets are organized as
    different columns.


    Parameters
    ==========

    estimation : numpy.ndarray (float)

      A N-dimensional array that corresponds to the value estimated by your
      procedure

    target : numpy.ndarray (float)

      A N-dimensional array that corresponds to the expected value


    Returns
    =======

    rmse : float

      The square-root of the average of the squared error between the estimated
      value and the target
    """
    return numpy.sqrt(mse(estimation, target))
