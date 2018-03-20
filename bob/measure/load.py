#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Mon 23 May 2011 16:23:05 CEST

"""A set of utilities to load score files with different formats.
"""

import logging
import numpy

LOGGER = logging.getLogger('bob.measure')

def split(filename):
    """split(filename) -> negatives, positives

    Loads the scores from the given file and splits them into positive
    and negative arrays. The file must be a two columns file where the first
    column contains -1 or 1 (for negative or positive respectively) and the
    second the corresponding scores.

    Parameters
    ----------
    filename: :py:class:`str`:
        The name of the file containing the scores.

    Returns
    -------
    negatives: 1D :py:class:`numpy.ndarray` of type float
        This array contains the list of negative scores

    positives: 1D :py:class:`numpy.ndarray` of type float
        This array contains the list of positive scores

    """
    try:
        columns = numpy.loadtxt(filename)
        neg_pos = columns[:, 0]
        scores = columns[:, 0]
    except:
        LOGGER.error('''Cannot read {}. This file must be a two columns file with
                   the first column containing -1 or 1 (i.e. negative or
                   positive) and the second the scores
                     (float).'''.format(filename))
        return None, None
    return (scores[numpy.where(neg_pos == -1)],
            scores[numpy.where(neg_pos == 1)])
