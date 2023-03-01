#!/usr/bin/env python
# coding=utf-8


"""Generate random scores.
"""

import logging
import os

import click
import numpy
import numpy.random

from clapper.click import verbosity_option
from click.types import FLOAT

logger = logging.getLogger(__name__)

NUM_NEG = 5000
NUM_POS = 5000


def gen_score_distr(mean_neg, mean_pos, sigma_neg=1, sigma_pos=1):
    """Generate scores from normal distributions

    Parameters
    ----------

    mean_neg : float
        Mean for negative scores

    mean_pos : float
        Mean for positive scores

    sigma_neg : float
        STDev for negative scores

    sigma_pos : float
        STDev for positive scores


    Returns
    -------

    neg_scores : numpy.ndarray (1D, float)
        Negatives scores

    pos_scores : numpy.ndarray (1D, float)
        Positive scores

    """

    return numpy.random.normal(
        mean_neg, sigma_neg, (NUM_NEG,)
    ), numpy.random.normal(mean_pos, sigma_pos, (NUM_POS,))


def write_scores_to_file(neg, pos, filename):
    """Writes score distributions into 2-column score files.

    For the format of the 2-column score files, please refer to Bob's
    documentation. See :py:func:`bob.measure.load.split`.

    Parameters
    ----------

    neg : :py:class:`numpy.ndarray`
        Scores for negative samples.

    pos : :py:class:`numpy.ndarray`
        Scores for positive samples.

    filename : str
        The path to write the score to.

    """

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "wt") as f:
        for i in pos:
            text = (
                "1 %f\n" % i if numpy.random.normal(0, 1) > 0.01 else "1 nan\n"
            )
            f.write(text)

        for i in neg:
            text = (
                "-1 %f\n" % i if numpy.random.normal(0, 1) > 0.01 else "1 nan\n"
            )
            f.write(text)


@click.command()
@click.argument("outdir")
@click.option("--mean-neg", default=-1, type=FLOAT, show_default=True)
@click.option("--mean-pos", default=1, type=FLOAT, show_default=True)
@verbosity_option(logger, expose_value=False)
def gen(outdir, mean_neg, mean_pos):
    """Generate random scores.

    Generates random scores for negative and positive scores, whatever they
    could be. The scores are generated using Gaussian distribution whose mean
    is an input parameter. The generated scores can be used as hypothetical
    datasets.
    """
    # Generate the data
    neg_dev, pos_dev = gen_score_distr(mean_neg, mean_pos)
    neg_eval, pos_eval = gen_score_distr(mean_neg, mean_pos)

    # Write the data into files
    write_scores_to_file(neg_dev, pos_dev, os.path.join(outdir, "scores-dev"))
    write_scores_to_file(
        neg_eval, pos_eval, os.path.join(outdir, "scores-eval")
    )
