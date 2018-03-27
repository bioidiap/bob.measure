'''Runs error analysis on score sets, outputs metrics and plots'''

from __future__ import division, print_function
import sys
import ntpath
import numpy
import click
import matplotlib
import matplotlib.pyplot as mpl
from matplotlib.backends.backend_pdf import PdfPages
from click.types import INT, FLOAT, Choice, File
from bob.extension.scripts.click_helper import verbosity_option
import bob.core
from .. import load
from .. import plot
from . import common_options
from tabulate import tabulate


def remove_nan(scores):
    """removes the NaNs from the scores"""
    nans = numpy.isnan(scores)
    sum_nans = sum(nans)
    total = len(scores)
    if sum_nans > 0:
        logger = bob.core.log.setup("bob.measure")
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

def _process_scores(dev_path, test_path, test):
    '''Process score files and return neg/pos/fta for test and dev'''
    dev_scores = load.split(dev_path)
    dev_neg = dev_pos = dev_fta = test_neg = test_pos = test_fta \
            = dev_file = test_file = None
    _, dev_file = ntpath.split(dev_path)
    if dev_scores[0] is not None:
        dev_scores, dev_fta = get_fta(dev_scores)
        dev_neg, dev_pos = dev_scores
        if dev_neg is None:
            raise click.UsageError("While loading dev-score file %s" %
                                   dev_path)

    if test and test_path is not None:
        test_scores = load.split(test_path)
        if test_scores[0] is not None:
            _, test_file = ntpath.split(test_path)
            test_scores, test_fta = get_fta(test_scores)
            test_neg, test_pos = test_scores
            if test_neg is None:
                raise click.UsageError("While loading test-score file %s" %\
                                       test_path)

    return (dev_neg, dev_pos, dev_fta, test_neg, test_pos,
            test_fta, dev_file, test_file)

def _get_thres(criter, neg, pos):
    '''Computes threshold from the given criterion and pos/neg scores'''
    if criter == 'eer':
        from .. import eer_threshold
        return eer_threshold(neg, pos)
    elif criter == 'hter':
        from .. import min_hter_threshold
        return min_hter_threshold(neg, pos)
    else:
        raise click.UsageError("Incorrect plotting criterion %s" % criter)

def _get_scores(ctx, scores=None):
    dev = scores if 'dev-scores' not in ctx.meta else ctx.meta['dev-scores']
    test = None if 'test-scores' not in ctx.meta else ctx.meta['test-scores']
    return (dev, test)

@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.table_option()
@common_options.test_option()
@common_options.open_file_mode_option()
@common_options.output_plot_metric_option()
@common_options.criterion_option()
@verbosity_option()
@click.pass_context
def metrics(ctx, criter, scores, log, test, open_mode,
            tablefmt='fancy_grid', **kargs):
    """Prints a single output line that contains all info for a given
    criterion (eer or hter).

    You need provide one or more development score file(s) for each experiment. 
    You can also provide test files along with dev files but the flag `--test`
    is required in that case.

    Resulting table format can be changer using the `--tablefmt`. Default
    formats are `fancy_grid` when output in the terminal and `latex` when
    written in a log file (see `--log`)

    Examples:
        $ bob measure metrics dev-scores

        $ bob measure metrics --test dev-scores1 test-scores1 dev-scores2
        test-scores2

        $ bob measure metrics --test -l results.txt dev-scores1 test-scores1
    """

    if log is not None:
        log_file = open(log, open_mode)
    else:
        log_file = sys.stdout

    dev_scores, test_scores = _get_scores(ctx, scores)
    for idx, dev_path in enumerate(dev_scores):
        test_path = test_scores[idx] if test_scores is not None else None
        dev_neg, dev_pos, dev_fta, test_neg, test_pos,\
        test_fta, dev_file, test_file =\
        _process_scores(dev_path, test_path, test)
        thres = _get_thres(criter, dev_neg, dev_pos)

        from .. import farfrr
        dev_fmr, dev_fnmr = farfrr(dev_neg, dev_pos, thres)
        dev_far = dev_fmr * (1 - dev_fta)
        dev_frr = dev_fta + dev_fnmr * (1 - dev_fta)
        dev_hter = (dev_far + dev_frr) / 2.0
        click.echo("[Min. criterion: %s] Threshold on Development set `%s`: %e"\
                   % (criter.upper(), dev_file, thres), file=log_file)

        dev_ni = dev_neg.shape[0]  # number of impostors
        dev_fm = int(round(dev_fmr * dev_ni))  # number of false accepts
        dev_nc = dev_pos.shape[0]  # number of clients
        dev_fnm = int(round(dev_fnmr * dev_nc))  # number of false rejects

        dev_fmr_str = "%.3f%% (%d/%d)" % (100 * dev_fmr, dev_fm, dev_ni)
        dev_fnmr_str = "%.3f%% (%d/%d)" % (100 * dev_fnmr, dev_fnm, dev_nc)
        dev_far_str = "%.3f%%" % (100 * dev_far)
        dev_frr_str = "%.3f%%" % (100 * dev_frr)
        dev_hter_str = "%.3f%%" % (100 * dev_hter)
        headers = ['', 'Development %s' % dev_file]
        raws = [['FMR', dev_fmr_str],
                ['FNMR', dev_fnmr_str],
                ['FAR', dev_far_str],
                ['FRR', dev_frr_str],
                ['HTER', dev_hter_str]]

        if test and test_neg is not None:
            # computes statistics for the test set based on the threshold a priori
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

            test_far_str = "%.3f%%" % (100 * test_far)
            test_frr_str = "%.3f%%" % (100 * test_frr)
            test_hter_str = "%.3f%%" % (100 * test_hter)

            headers.append('Test % s' % test_file)
            raws[0].append(test_fmr_str)
            raws[1].append(test_fnmr_str)
            raws[2].append(test_far_str)
            raws[3].append(test_frr_str)
            raws[4].append(test_hter_str)

        click.echo(tabulate(raws, headers, tablefmt), file=log_file)
    if log is not None:
        log_file.close()

def _mplt_setup(ctx, output):
    if not hasattr(matplotlib, 'backends'):
        matplotlib.use('pdf')

    if 'PdfPages' not in ctx.meta:
        ctx.meta['PdfPages'] = PdfPages(output)
    return ctx.meta['PdfPages']

def _end_pp(ctx, pp):
    if 'PdfPages' not in ctx.meta:
        return
    if 'closef' not in ctx.meta or ctx.meta['closef']:
        pp.close()

@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.output_plot_file_option(default_out='roc.pdf')
@common_options.test_option()
@common_options.points_curve_option()
@verbosity_option()
@click.pass_context
def roc(ctx, output, scores, test, points, **kargs):
    """Plot ROC (receiver operating characteristic) curve:
    plot of the rate of false positives (i.e. impostor attempts accepted) on the
    x-axis against the corresponding rate of true positives (i.e. genuine attempts
    accepted) on the y-axis plotted parametrically as a function of a decision
    threshold

    You need provide one or more development score file(s) for each experiment.
    You can also provide test files along with dev files but the flag `--test`
    is required in that case.

    Examples:
        $ bob measure roc dev-scores

        $ bob measure roc --test dev-scores1 test-scores1 dev-scores2
        test-scores2

        $ bob measure roc --test -o my_roc.pdf dev-scores1 test-scores1
    """
    pp = _mplt_setup(ctx, output)

    dev_scores, test_scores = _get_scores(ctx, scores)
    for idx, dev_path in enumerate(dev_scores):
        test_path = test_scores[idx] if test_scores is not None else None
        dev_neg, dev_pos, _, test_neg, test_pos,\
        _, dev_file, test_file =\
        _process_scores(dev_path, test_path, test)

        fig = mpl.figure()
        if test:
            plot.roc(dev_neg, dev_pos, points, color=(0.3, 0.3, 0.3),
                     linestyle='--', dashes=(6, 2), label='development')
            plot.roc(test_neg, test_pos, points, color=(0, 0, 0),
                     linestyle='-', label='test')
        else:
            plot.roc(dev_neg, dev_pos, points, color=(0, 0, 0),
                     linestyle='-', label='development')

        title = dev_file + (" / %s" % test_file if test else "")
        mpl.axis([0, 40, 0, 40])
        mpl.title("ROC Curve (%s)" % title)
        mpl.xlabel('FMR (%)')
        mpl.ylabel('FNMR (%)')
        mpl.grid(True, color=(0.3, 0.3, 0.3))
        if test:
            mpl.legend()
        pp.savefig(fig)
    _end_pp(ctx, pp)

@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.output_plot_file_option(default_out='det.pdf')
@common_options.test_option()
@common_options.points_curve_option()
@verbosity_option()
@click.pass_context
def det(ctx, output, scores, test, points, **kargs):
    """Plot DET (detection error trade-off) curve:
    modified ROC curve which plots error rates on both axes
    (false positives on the x-axis and false negatives on the y-axis)

    You need provide one or more development score file(s) for each experiment.
    You can also provide test files along with dev files but the flag `--test`
    is required in that case.

    Examples:
        $ bob measure det dev-scores

        $ bob measure det --test dev-scores1 test-scores1 dev-scores2
        test-scores2

        $ bob measure det --test -o my_det.pdf dev-scores1 test-scores1
    """
    pp = _mplt_setup(ctx, output)

    dev_scores, test_scores = _get_scores(ctx, scores)
    for idx, dev_path in enumerate(dev_scores):
        test_path = test_scores[idx] if test_scores is not None else None
        dev_neg, dev_pos, _, test_neg, test_pos,\
        _, dev_file, test_file =\
        _process_scores(dev_path, test_path, test)
        fig = mpl.figure()
        if test and test_neg is not None:
            plot.det(dev_neg, dev_pos, points, color=(0.3, 0.3, 0.3),
                     linestyle='--', dashes=(6, 2), label='development')
            plot.det(test_neg, test_pos, points, color=(0, 0, 0),
                     linestyle='-', label='test')
        else:
            plot.det(dev_neg, dev_pos, points, color=(0, 0, 0),
                     linestyle='-', label='development')

        plot.det_axis([0.01, 40, 0.01, 40])
        title = dev_file + (" / %s" % test_file if test else "")
        mpl.title("DET Curve (%s)" % title)
        mpl.xlabel('FMR (%)')
        mpl.ylabel('FNMR (%)')
        mpl.grid(True, color=(0.3, 0.3, 0.3))
        if test:
            mpl.legend()
        pp.savefig(fig)
    _end_pp(ctx, pp)

@click.command()
@common_options.scores_argument(test_mandatory=True, nargs=-1)
@common_options.output_plot_file_option(default_out='epc.pdf')
@common_options.points_curve_option()
@verbosity_option()
@click.pass_context
def epc(ctx, output, points, **kargs):
    """Plot EPC (expected performance curve):
    plots the error rate on the test set depending on a threshold selected
    a-priori on the development set and accounts for varying relative cost β
    ∈ [0; 1] of FPR and FNR when calculating the threshold.

    You need provide one or more development score and test file(s)
    for each experiment.

    Examples:
        $ bob measure epc dev-scores test-scores

        $ bob measure epc -o my_epc.pdf dev-scores1 test-scores1
    """
    pp = _mplt_setup(ctx, output)

    if 'dev-scores' not in ctx.meta or 'test-scores' not in ctx.meta:
        raise click.UsageError("EPC requires dev and test score files")

    dev_scores, test_scores = _get_scores(ctx)
    for dev_path, test_path in zip(dev_scores, test_scores):
        dev_neg, dev_pos, _, test_neg, test_pos,\
        _, dev_file, test_file =\
        _process_scores(dev_path, test_path, True)
        fig = mpl.figure()
        plot.epc(dev_neg, dev_pos, test_neg, test_pos, points,
                 color=(0, 0, 0), linestyle='-')
        title = dev_file + " / "  + test_file
        mpl.title('EPC Curve (%s)' % title)
        mpl.xlabel('Cost')
        mpl.ylabel('Min. HTER (%)')
        mpl.grid(True, color=(0.3, 0.3, 0.3))
        pp.savefig(fig)
    _end_pp(ctx, pp)

@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.output_plot_file_option(default_out='hist.pdf')
@common_options.test_option()
@common_options.n_bins_option()
@common_options.criterion_option()
@verbosity_option()
@click.pass_context
def hist(ctx, output, scores, criter, test, nbins, **kargs):
    """ Plots histograms of positive and negatives along with threshold
    criterion.

    You need provide one or more development score file(s) for each experiment.
    You can also provide test files along with dev files but the flag `--test`
    is required in that case.

    Examples:
        $ bob measure hist dev-scores

        $ bob measure hist --test dev-scores1 test-scores1 dev-scores2
        test-scores2

        $ bob measure hist --test --criter hter dev-scores1 test-scores1
    """
    pp = _mplt_setup(ctx, output)

    dev_scores, test_scores = _get_scores(ctx, scores)
    for idx, dev_path in enumerate(dev_scores):
        test_path = test_scores[idx] if test_scores is not None else None
        dev_neg, dev_pos, _, test_neg, test_pos,\
        _, dev_file, test_file =\
        _process_scores(dev_path, test_path, test)
        thres = _get_thres(criter, dev_neg, dev_pos)
        fig = mpl.figure()
        if test_neg is not None:
            mpl.subplot(2, 1, 1)
            all_scores = numpy.hstack((dev_neg, test_neg, dev_pos, test_pos))
        else:
            all_scores = numpy.hstack((dev_neg, dev_pos))
        score_range = all_scores.min(), all_scores.max()

        def setup_hist(neg, pos, xlim, thres, y_label, x_label='Score values'):
            mpl.hist(neg, label='Positives', normed=True, color='red',
                     alpha=0.5, bins=nbins)
            mpl.hist(pos, label='Negatives', normed=True, color='blue',
                     alpha=0.5, bins=nbins)
            mpl.xlim(*xlim)
            _, _, ymax, ymin = mpl.axis()
            mpl.vlines(thres, ymin, ymax, color='black',
                       label=criter.upper(), linestyle='dashed')
            mpl.grid(True, alpha=0.5)
            mpl.ylabel(y_label)
            if x_label is not None:
                mpl.xlabel(x_label)

        title = dev_file + (" / %s" % test_file if test else "")
        mpl.title('Score Distributions (%s)' % title)
        y_label = 'Dev. Scores (normalized)' if test else 'Normalized Count'
        x_label = 'Score values' if not test else ''
        setup_hist(dev_neg, dev_pos, score_range, thres, y_label, x_label)
        if test_neg is not None:
            ax = mpl.gca()
            ax.axes.get_xaxis().set_ticklabels([])
            mpl.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, -0.01),
                       fontsize=10)
        else:
            mpl.legend(loc='best', fancybox=True, framealpha=0.5)

        if test_neg is not None:
            mpl.subplot(2, 1, 2)
            setup_hist(test_neg, test_pos, score_range, thres,
                       'Test Scores (normalized)')
        pp.savefig(fig)
    _end_pp(ctx, pp)

@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.table_option()
@common_options.test_option()
@common_options.output_plot_metric_option()
@common_options.output_plot_file_option(default_out='eval_plots.pdf')
@common_options.points_curve_option()
@common_options.n_bins_option()
@verbosity_option()
@click.pass_context
def evaluate(ctx, scores, tablefmt, test, log, **kargs):
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
        $ bob measure evaluate dev-scores

        $ bob measure evaluate -t -l metrics.txt -o my_plots.pdf dev-scores test-scores
    '''
    #first time erase if existing file
    click.echo("Computing metrics with EER...")
    ctx.invoke(metrics, criter='eer', scores=scores, log=log,
               test=test, open_mode='w', tablefmt=tablefmt)
    #second time, appends the content
    click.echo("Computing metrics with HTER...")
    ctx.invoke(metrics, criter='hter', scores=scores, log=log,
               test=test, open_mode='a', tablefmt=tablefmt)

    #avoid closing pdf file before all figures are plotted
    ctx.meta['closef'] = False
    if test:
        click.echo("Starting evaluate with dev and test scores...")
    else:
        click.echo("Starting evaluate with dev scores only...")
    click.echo("Computing ROC...")
    ctx.forward(roc)
    click.echo("Computing DET...")
    ctx.forward(det)
    if test:
        click.echo("Computing EPC...")
        ctx.forward(epc)
    #the last one closes the file
    ctx.meta['closef'] = True
    click.echo("Computing score histograms...")
    ctx.forward(hist)
    click.echo("Evaluate successfully completed!")
    return 0
