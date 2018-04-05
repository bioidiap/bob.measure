''' Click commands for ``bob.measure`` '''


import click
from .. import load
from . import figure
from . import common_options
from bob.extension.scripts.click_helper import verbosity_option


@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.test_option()
@common_options.table_option()
@common_options.open_file_mode_option()
@common_options.output_plot_metric_option()
@common_options.criterion_option()
@common_options.threshold_option()
@verbosity_option()
@click.pass_context
def metrics(ctx, scores, test, **kwargs):
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

        $ bob measure metrics --test -l results.txt dev-scores1 test-scores1

        $ bob measure metrics --test {dev,test}-scores1 {dev,test}-scores2
    """
    process = figure.Metrics(ctx, scores, test, load.split_files)
    process.run()


@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.titles_option()
@common_options.sep_dev_test_option()
@common_options.output_plot_file_option(default_out='roc.pdf')
@common_options.test_option()
@common_options.points_curve_option()
@common_options.semilogx_option(True)
@common_options.axes_val_option(dflt=[1e-4, 1, 1e-4, 1])
@common_options.axis_fontsize_option()
@common_options.x_rotation_option()
@common_options.fmr_line_at_option()
@verbosity_option()
@click.pass_context
def roc(ctx, scores, test, **kwargs):
    """Plot ROC (receiver operating characteristic) curve:
    The plot will represent the false match rate on the horizontal axis and the
    false non match rate on the vertical axis.  The values for the axis will be
    computed using :py:func:`bob.measure.roc`.

    You need provide one or more development score file(s) for each experiment.
    You can also provide test files along with dev files but the flag `--test`
    is required in that case.

    Examples:
        $ bob measure roc dev-scores

        $ bob measure roc --test dev-scores1 test-scores1 dev-scores2
        test-scores2

        $ bob measure roc --test -o my_roc.pdf dev-scores1 test-scores1
    """
    process = figure.Roc(ctx, scores, test, load.split_files)
    process.run()


@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.output_plot_file_option(default_out='det.pdf')
@common_options.titles_option()
@common_options.sep_dev_test_option()
@common_options.test_option()
@common_options.axes_val_option(dflt=[0.01, 95, 0.01, 95])
@common_options.axis_fontsize_option(dflt=6)
@common_options.x_rotation_option(dflt=45)
@common_options.points_curve_option()
@verbosity_option()
@click.pass_context
def det(ctx, scores, test, **kwargs):
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
    process = figure.Det(ctx, scores, test, load.split_files)
    process.run()


@click.command()
@common_options.scores_argument(test_mandatory=True, nargs=-1)
@common_options.output_plot_file_option(default_out='epc.pdf')
@common_options.titles_option()
@common_options.points_curve_option()
@common_options.axis_fontsize_option()
@verbosity_option()
@click.pass_context
def epc(ctx, scores, **kwargs):
    """Plot EPC (expected performance curve):
    plots the error rate on the test set depending on a threshold selected
    a-priori on the development set and accounts for varying relative cost
    in [0; 1] of FPR and FNR when calculating the threshold.

    You need provide one or more development score and test file(s)
    for each experiment.

    Examples:
        $ bob measure epc dev-scores test-scores

        $ bob measure epc -o my_epc.pdf dev-scores1 test-scores1
    """
    process = figure.Epc(ctx, scores, True, load.split_files)
    process.run()


@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.output_plot_file_option(default_out='hist.pdf')
@common_options.test_option()
@common_options.n_bins_option()
@common_options.criterion_option()
@common_options.axis_fontsize_option()
@common_options.threshold_option()
@verbosity_option()
@click.pass_context
def hist(ctx, scores, test, **kwargs):
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
    process = figure.Hist(ctx, scores, test, load.split_files)
    process.run()


@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.titles_option()
@common_options.sep_dev_test_option()
@common_options.table_option()
@common_options.test_option()
@common_options.output_plot_metric_option()
@common_options.output_plot_file_option(default_out='eval_plots.pdf')
@common_options.points_curve_option()
@common_options.semilogx_option(dflt=True)
@common_options.n_bins_option()
@common_options.fmr_line_at_option()
@verbosity_option()
@click.pass_context
def evaluate(ctx, scores, test, **kwargs):
    '''Runs error analysis on score sets

    \b
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
    # first time erase if existing file
    click.echo("Computing metrics with EER...")
    ctx.meta['criter'] = 'eer'  # no criterion passed to evaluate
    ctx.invoke(metrics, scores=scores, test=test)
    # second time, appends the content
    click.echo("Computing metrics with HTER...")
    ctx.meta['criter'] = 'hter'  # no criterion passed in evaluate
    ctx.invoke(metrics, scores=scores, test=test)
    if 'log' in ctx.meta:
        click.echo("[metrics] => %s" % ctx.meta['log'])

    # avoid closing pdf file before all figures are plotted
    ctx.meta['closef'] = False
    if test:
        click.echo("Starting evaluate with dev and test scores...")
    else:
        click.echo("Starting evaluate with dev scores only...")
    click.echo("Computing ROC...")
    # set axes limits for ROC
    ctx.forward(roc) # use class defaults plot settings
    click.echo("Computing DET...")
    ctx.forward(det) # use class defaults plot settings
    if test:
        click.echo("Computing EPC...")
<<<<<<< HEAD
        ctx.forward(epc) # use class defaults plot settings
    # the last one closes the file
||||||| merged common ancestors
        ctx.forward(epc)
<<<<<<< HEAD
    #the last one closes the file
=======
        ctx.forward(epc)
    # the last one closes the file
>>>>>>> b98021d93c81eee46a730271fad67e001bf084ac
||||||| merged common ancestors
    #the last one closes the file
=======
    # the last one closes the file
>>>>>>> b98021d93c81eee46a730271fad67e001bf084ac
    ctx.meta['closef'] = True
    click.echo("Computing score histograms...")
    ctx.forward(hist)
    click.echo("Evaluate successfully completed!")
    click.echo("[plots] => %s" % (ctx.meta['output']))
