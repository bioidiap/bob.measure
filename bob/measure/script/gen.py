"""Generate random scores.
"""
import os
import logging
import numpy
import click
from click.types import FLOAT
from bob.extension.scripts.click_helper import verbosity_option
from bob.core import random
from bob.io.base import create_directories_safe

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
    neg_scores : :any:`list`
        Negatives scores
    pos_scores : :any:`list`
        Positive scores
    """
    mt = random.mt19937()  # initialise the random number generator

    neg_generator = random.normal(numpy.float32, mean_neg, sigma_neg)
    pos_generator = random.normal(numpy.float32, mean_pos, sigma_pos)

    neg_scores = [neg_generator(mt) for _ in range(NUM_NEG)]
    pos_scores = [pos_generator(mt) for _ in range(NUM_NEG)]

    return neg_scores, pos_scores


def write_scores_to_file(neg, pos, filename):
    """Writes score distributions into 2-column score files. For the format of
    the 2-column score files, please refer to Bob's documentation. See
    :py:func:`bob.measure.load.split`.

    Parameters
    ----------
    neg : :py:class:`numpy.ndarray`
        Scores for negative samples.
    pos : :py:class:`numpy.ndarray`
        Scores for positive samples.
    filename : str
        The path to write the score to.
    """
    create_directories_safe(os.path.dirname(filename))
    mt = random.mt19937()
    nan_dist = random.uniform(numpy.float32, 0, 1)
    with open(filename, 'wt') as f:
        for i in pos:
            text = '1 %f\n' % i if nan_dist(mt) > 0.01 else '1 nan\n'
            f.write(text)
        for i in neg:
            text = '-1 %f\n' % i if nan_dist(mt) > 0.01 else '1 nan\n'
            f.write(text)


@click.command()
@click.argument('outdir')
@click.option('--mean-neg', default=-1, type=FLOAT, show_default=True)
@click.option('--mean-pos', default=1, type=FLOAT, show_default=True)
@verbosity_option()
def gen(outdir, mean_neg, mean_pos, **kwargs):
    """Generate random scores.
    Generates random scores for negative and positive scores, whatever they
    could be. The
    scores are generated using Gaussian distribution whose mean is an input
    parameter. The generated scores can be used as hypothetical datasets.
    """
    # Generate the data
    neg_dev, pos_dev = gen_score_distr(mean_neg, mean_pos)
    neg_eval, pos_eval = gen_score_distr(mean_neg, mean_pos)

    # Write the data into files
    write_scores_to_file(neg_dev, pos_dev,
                         os.path.join(outdir, 'scores-dev'))
    write_scores_to_file(neg_eval, pos_eval,
                         os.path.join(outdir, 'scores-eval'))
