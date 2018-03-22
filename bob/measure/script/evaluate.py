'''Runs error analysis on score sets, outputs metrics and plots'''

from __future__ import division, print_function
import sys
import numpy
import click
from click.types import INT, FLOAT, Choice, File
import bob.core
from  .common_options import (plot_options, normalize_options)
from .. import load

LOG_FILE = sys.stdout

logger = bob.core.log.setup("bob.measure")

def remove_nan(scores):
    """removes the NaNs from the scores"""
    nans = numpy.isnan(scores)
    sum_nans = sum(nans)
    total = len(scores)
    if sum_nans > 0:
        logger.warning('Found {} NaNs in {} scores'.format(sum_nans, total))
    return scores[numpy.where(~nans)], sum_nans, total


def get_fta(scores):
    """calculates the Failure To Acquire (FtA) rate"""
    fta_sum, fta_total = 0, 0
    neg, sum_nans, total = remove_nan(scores[0])
    fta_sum += sum_nans
    fta_total += total
    pos, sum_nans, total = remove_nan(scores[1])
    fta_sum += sum_nans
    fta_total += total
    return ((neg, pos), fta_sum / fta_total)



def print_crit(crit, dev_scores, dev_fta, test_scores=None, test_fta=None):
    """Prints a single output line that contains all info for a given criterion"""

    dev_neg, dev_pos = dev_scores

    if crit == 'EER':
        from .. import eer_threshold
        thres = eer_threshold(dev_neg, dev_pos)
    else:
        from .. import min_hter_threshold
        thres = min_hter_threshold(dev_neg, dev_pos)

    from .. import farfrr
    dev_fmr, dev_fnmr = farfrr(dev_neg, dev_pos, thres)
    dev_far = dev_fmr * (1 - dev_fta)
    dev_frr = dev_fta + dev_fnmr * (1 - dev_fta)
    dev_hter = (dev_far + dev_frr) / 2.0

    print("[Min. criterion: %s] Threshold on Development set: %e" % (crit, thres), file=LOG_FILE)

    dev_ni = dev_neg.shape[0]  # number of impostors
    dev_fm = int(round(dev_fmr * dev_ni))  # number of false accepts
    dev_nc = dev_pos.shape[0]  # number of clients
    dev_fnm = int(round(dev_fnmr * dev_nc))  # number of false rejects

    dev_fmr_str = "%.3f%% (%d/%d)" % (100 * dev_fmr, dev_fm, dev_ni)
    dev_fnmr_str = "%.3f%% (%d/%d)" % (100 * dev_fnmr, dev_fnm, dev_nc)
    dev_max_len = max(len(dev_fmr_str), len(dev_fnmr_str))

    def fmt(s, space):
        return ('%' + ('%d' % space) + 's') % s

    if test_scores is None:
        # prints only dev performance rates
        print("       | %s" % fmt("Development", -1 * dev_max_len), file=LOG_FILE)
        print("-------+-%s" % (dev_max_len * "-"), file=LOG_FILE)
        print("  FMR  | %s" % fmt(dev_fmr_str, -1 * dev_max_len), file=LOG_FILE)
        print("  FNMR | %s" % fmt(dev_fnmr_str, -1 * dev_max_len), file=LOG_FILE)
        dev_far_str = "%.3f%%" % (100 * dev_far)
        print("  FAR  | %s" % fmt(dev_far_str, -1 * dev_max_len), file=LOG_FILE)
        dev_frr_str = "%.3f%%" % (100 * dev_frr)
        print("  FRR  | %s" % fmt(dev_frr_str, -1 * dev_max_len), file=LOG_FILE)
        dev_hter_str = "%.3f%%" % (100 * dev_hter)
        print("  HTER | %s" % fmt(dev_hter_str, -1 * dev_max_len), file=LOG_FILE)
    else:
        # computes statistics for the test set based on the threshold a priori
        test_neg, test_pos = test_scores

        test_fmr, test_fnmr = farfrr(test_neg, test_pos, thres)
        test_far = test_fmr * (1 - test_fta)
        test_frr = test_fta + test_fnmr * (1 - test_fta)
        test_hter = (test_far + test_frr) / 2.0

        test_ni = test_neg.shape[0]  # number of impostors
        test_fm = int(round(test_fmr * test_ni))  # number of false accepts
        test_nc = test_pos.shape[0]  # number of clients
        test_fnm = int(round(test_fnmr * test_nc))  # number of false rejects

        test_fmr_str = "%.3f%% (%d/%d)" % (100 * test_fmr, test_fm, test_ni)
        test_fnmr_str = "%.3f%% (%d/%d)" % (100 * test_fnmr, test_fnm, test_nc)
        test_max_len = max(len(test_fmr_str), len(test_fnmr_str))

        # prints both dev and test performance rates
        print("       | %s | %s" % (fmt("Development", -1 * dev_max_len),
                                    fmt("Test", -1 * test_max_len)),
              file=LOG_FILE)
        print("-------+-%s-+-%s" % (dev_max_len * "-", (2 + test_max_len) * "-"),
              file=LOG_FILE)
        print("  FMR  | %s | %s" % (fmt(dev_fmr_str, -1 * dev_max_len),
                                    fmt(test_fmr_str, -1 * test_max_len)),
              file=LOG_FILE)
        print("  FNMR | %s | %s" % (fmt(dev_fnmr_str, -1 * dev_max_len),
                                    fmt(test_fnmr_str, -1 * test_max_len)),
              file=LOG_FILE)
        dev_far_str = "%.3f%%" % (100 * dev_far)
        test_far_str = "%.3f%%" % (100 * test_far)
        print("  FAR  | %s | %s" % (fmt(dev_far_str, -1 * dev_max_len),
                                    fmt(test_far_str, -1 * test_max_len)),
              file=LOG_FILE)
        dev_frr_str = "%.3f%%" % (100 * dev_frr)
        test_frr_str = "%.3f%%" % (100 * test_frr)
        print(
            "  FRR  | %s | %s" % (fmt(dev_frr_str, -1 * dev_max_len),
                                  fmt(test_frr_str, -1 * test_max_len)),
            file=LOG_FILE
        )
        dev_hter_str = "%.3f%%" % (100 * dev_hter)
        test_hter_str = "%.3f%%" % (100 * test_hter)
        print(
            "  HTER | %s | %s" % (fmt(dev_hter_str, -1 * dev_max_len),
                                  fmt(test_hter_str, -1 * test_max_len)),
            file=LOG_FILE
        )


def plots(crit, points, filename, dev_scores, test_scores=None):
    """Saves ROC, DET and EPC curves on the file pointed out by filename."""

    dev_neg, dev_pos = dev_scores

    if test_scores is not None:
        test_neg, test_pos = test_scores
    else:
        test_neg, test_pos = None, None

    from .. import plot

    import matplotlib
    if not hasattr(matplotlib, 'backends'):
        matplotlib.use('pdf')
    import matplotlib.pyplot as mpl
    from matplotlib.backends.backend_pdf import PdfPages

    pp = PdfPages(filename)

    # ROC
    fig = mpl.figure()

    if test_scores is not None:
        plot.roc(dev_neg, dev_pos, points, color=(0.3, 0.3, 0.3),
                 linestyle='--', dashes=(6, 2), label='development')
        plot.roc(test_neg, test_pos, points, color=(0, 0, 0),
                 linestyle='-', label='test')
    else:
        plot.roc(dev_neg, dev_pos, points, color=(0, 0, 0),
                 linestyle='-', label='development')

    mpl.axis([0, 40, 0, 40])
    mpl.title("ROC Curve")
    mpl.xlabel('FMR (%)')
    mpl.ylabel('FNMR (%)')
    mpl.grid(True, color=(0.3, 0.3, 0.3))
    if test_scores is not None:
        mpl.legend()
    pp.savefig(fig)

    # DET
    fig = mpl.figure()

    if test_scores is not None:
        plot.det(dev_neg, dev_pos, points, color=(0.3, 0.3, 0.3),
                 linestyle='--', dashes=(6, 2), label='development')
        plot.det(test_neg, test_pos, points, color=(0, 0, 0),
                 linestyle='-', label='test')
    else:
        plot.det(dev_neg, dev_pos, points, color=(0, 0, 0),
                 linestyle='-', label='development')

    plot.det_axis([0.01, 40, 0.01, 40])
    mpl.title("DET Curve")
    mpl.xlabel('FMR (%)')
    mpl.ylabel('FNMR (%)')
    mpl.grid(True, color=(0.3, 0.3, 0.3))
    if test_scores is not None:
        mpl.legend()
    pp.savefig(fig)

    # EPC - requires test set
    if test_scores is not None:
        fig = mpl.figure()
        plot.epc(dev_neg, dev_pos, test_neg, test_pos, points,
                 color=(0, 0, 0), linestyle='-')
        mpl.title('EPC Curve')
        mpl.xlabel('Cost')
        mpl.ylabel('Min. HTER (%)')
        mpl.grid(True, color=(0.3, 0.3, 0.3))
        pp.savefig(fig)

    # Distribution for dev and test scores on the same page
    if crit == 'EER':
        from .. import eer_threshold
        thres = eer_threshold(dev_neg, dev_pos)
    else:
        from .. import min_hter_threshold
        thres = min_hter_threshold(dev_neg, dev_pos)

    fig = mpl.figure()

    if test_scores is not None:
        mpl.subplot(2, 1, 1)
        all_scores = numpy.hstack((dev_neg, test_neg, dev_pos, test_pos))
    else:
        all_scores = numpy.hstack((dev_neg, dev_pos))

    nbins = 20
    score_range = all_scores.min(), all_scores.max()
    mpl.hist(dev_neg, label='Impostors', normed=True, color='red', alpha=0.5,
             bins=nbins)
    mpl.hist(dev_pos, label='Genuine', normed=True, color='blue', alpha=0.5,
             bins=nbins)
    mpl.xlim(*score_range)
    _, _, ymax, ymin = mpl.axis()
    mpl.vlines(thres, ymin, ymax, color='black', label='EER', linestyle='dashed')

    if test_scores is not None:
        ax = mpl.gca()
        ax.axes.get_xaxis().set_ticklabels([])
        mpl.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, -0.01),
                  fontsize=10)
        mpl.ylabel('Dev. Scores (normalized)')
    else:
        mpl.ylabel('Normalized Count')
        mpl.legend(loc='best', fancybox=True, framealpha=0.5)
    mpl.title('Score Distributions')
    mpl.grid(True, alpha=0.5)

    if test_scores is not None:
        mpl.subplot(2, 1, 2)
        mpl.hist(test_neg, label='Impostors', normed=True, color='red', alpha=0.5,
                 bins=nbins)
        mpl.hist(test_pos, label='Genuine', normed=True, color='blue', alpha=0.5,
                 bins=nbins)
        mpl.ylabel('Test Scores (normalized)')
        mpl.xlabel('Score value')
        mpl.xlim(*score_range)
        _, _, ymax, ymin = mpl.axis()
        mpl.vlines(thres, ymin, ymax, color='black', label='EER',
                   linestyle='dashed')
        mpl.grid(True, alpha=0.5)

    pp.savefig(fig)

    pp.close()

@click.command()
@click.option('-l', '--log', help='If provided, computed numbers are written to \
              this file instead of the standard output.')
@click.option('-x', '--no-plot', default=False, show_default=True,
              help='If True, then I\'ll execute no plotting')
@click.option('-n', '--points', type=INT, default=100, show_default=True,
              help='Number of points to use in the curves')
@click.option('-o', '--output', default='curves.pdf', show_default=True,
              help='Number of points to use in the curves')
@click.argument('scores', nargs=-1)
def evaluate(scores, output, points, no_plot, log):
    '''Runs error analysis on score sets
    1. Computes the threshold using either EER or min. HTER criteria on
        development set scores
    2. Applies the above threshold on test set scores to compute the HTER, if a
        test-score set is provided
    3. Reports error rates on the console
    4. Plots ROC, EPC, DET curves and score distributions to a multi-page PDF
        file (unless --no-plot is passed)


    You need to provide 2 score files for each biometric system in this order:
    \b
    * development scores
    * evaluation scores

    Examples:
        $ bob measure evaluate dev-scores test-scores

    '''

    if len(scores) < 1:
        logger.error("No score argument(s).")
        return -1

    # setup the logfile
    global LOG_FILE
    if log is not None:
        LOG_FILE = open(log, 'w')

    assert points > 0, "Numbers of points must be positive"

    dev_scores = load.split(scores[0])
    if dev_scores[0] is None:
        logger.error("While loading dev-score file")
        return -1

    if len(scores) > 1:
        test_scores = load.split(scores[1])
        if test_scores[1] is None:
            logger.error("While loading test-score file")
    else:
        test_scores = None
        test_fta = None

    # test if there are nan in the score files and remove them
    # also calculate FTA
    dev_scores, dev_fta = get_fta(dev_scores)
    print("Failure To Acquire (FTA) in the development set is: {:.3f}%".format(
        dev_fta * 100), file=LOG_FILE)
    if test_scores is not None:
        test_scores, test_fta = get_fta(test_scores)
        print("Failure To Acquire (FTA) in the test set is: {:.3f}%".format(
            test_fta * 100), file=LOG_FILE)

    print_crit('EER', dev_scores, dev_fta, test_scores, test_fta)
    print_crit('Min. HTER', dev_scores, dev_fta, test_scores, test_fta)

    if not no_plot:
        plots(
            'EER', points,
            output, dev_scores,
            test_scores
        )
        print(
            "[Plots] Performance curves => '%s'" % output,
            file=LOG_FILE
        )

    LOG_FILE.flush()
    return 0
