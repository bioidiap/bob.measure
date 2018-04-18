''' Click commands for ``bob.measure`` '''


import click
from .. import load
from . import figure
from . import common_options
from bob.extension.scripts.click_helper import (verbosity_option,
                                                open_file_mode_option)

@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.eval_option()
@common_options.table_option()
@common_options.output_plot_metric_option()
@common_options.criterion_option()
@common_options.thresholds_option()
@common_options.far_option()
@common_options.titles_option()
@open_file_mode_option()
@verbosity_option()
@click.pass_context
def metrics(ctx, scores, evaluation, **kwargs):
    """Prints a table that contains FtA, FAR, FRR, FMR, FMNR, HTER for a given
    threshold criterion (eer or hter).

    You need to provide one or more development score file(s) for each experiment.
    You can also provide evaluation files along with dev files. If only dev scores
    are provided, you must use flag `--no-evaluation`.

    Resulting table format can be changed using the `--tablefmt`.

    Examples:
        $ bob measure metrics dev-scores

        $ bob measure metrics -l results.txt dev-scores1 eval-scores1

        $ bob measure metrics {dev,eval}-scores1 {dev,eval}-scores2
    """
    process = figure.Metrics(ctx, scores, evaluation, load.split_files)
    process.run()

@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.title_option()
@common_options.titles_option()
@common_options.sep_dev_eval_option()
@common_options.output_plot_file_option(default_out='roc.pdf')
@common_options.eval_option()
@common_options.points_curve_option()
@common_options.semilogx_option(True)
@common_options.axes_val_option(dflt=[1e-4, 1, 1e-4, 1])
@common_options.x_rotation_option()
@common_options.x_label_option()
@common_options.y_label_option()
@common_options.lines_at_option()
@common_options.const_layout_option()
@common_options.figsize_option()
@common_options.style_option()
@common_options.const_layout_option()
@verbosity_option()
@click.pass_context
def roc(ctx, scores, evaluation, **kwargs):
    """Plot ROC (receiver operating characteristic) curve:
    The plot will represent the false match rate on the horizontal axis and the
    false non match rate on the vertical axis.  The values for the axis will be
    computed using :py:func:`bob.measure.roc`.

    You need to provide one or more development score file(s) for each experiment.
    You can also provide evaluation files along with dev files. If only dev scores
    are provided, you must use flag `--no-evaluation`.

    Examples:
        $ bob measure roc dev-scores

        $ bob measure roc dev-scores1 eval-scores1 dev-scores2
        eval-scores2

        $ bob measure roc -o my_roc.pdf dev-scores1 eval-scores1
    """
    process = figure.Roc(ctx, scores, evaluation, load.split_files)
    process.run()

@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.output_plot_file_option(default_out='det.pdf')
@common_options.title_option()
@common_options.titles_option()
@common_options.sep_dev_eval_option()
@common_options.eval_option()
@common_options.axes_val_option(dflt=[0.01, 95, 0.01, 95])
@common_options.x_rotation_option(dflt=45)
@common_options.x_label_option()
@common_options.y_label_option()
@common_options.points_curve_option()
@common_options.const_layout_option()
@common_options.figsize_option()
@common_options.style_option()
@common_options.const_layout_option()
@verbosity_option()
@click.pass_context
def det(ctx, scores, evaluation, **kwargs):
    """Plot DET (detection error trade-off) curve:
    modified ROC curve which plots error rates on both axes
    (false positives on the x-axis and false negatives on the y-axis)

    You need to provide one or more development score file(s) for each experiment.
    You can also provide evaluation files along with dev files. If only dev scores
    are provided, you must use flag `--no-evaluation`.

    Examples:
        $ bob measure det dev-scores

        $ bob measure det dev-scores1 eval-scores1 dev-scores2
        eval-scores2

        $ bob measure det -o my_det.pdf dev-scores1 eval-scores1
    """
    process = figure.Det(ctx, scores, evaluation, load.split_files)
    process.run()

@click.command()
@common_options.scores_argument(eval_mandatory=True, nargs=-1)
@common_options.output_plot_file_option(default_out='epc.pdf')
@common_options.title_option()
@common_options.titles_option()
@common_options.points_curve_option()
@common_options.const_layout_option()
@common_options.x_label_option()
@common_options.y_label_option()
@common_options.figsize_option()
@common_options.style_option()
@common_options.const_layout_option()
@verbosity_option()
@click.pass_context
def epc(ctx, scores, **kwargs):
    """Plot EPC (expected performance curve):
    plots the error rate on the eval set depending on a threshold selected
    a-priori on the development set and accounts for varying relative cost
    in [0; 1] of FPR and FNR when calculating the threshold.

    You need to provide one or more development score and eval file(s)
    for each experiment.

    Examples:
        $ bob measure epc dev-scores eval-scores

        $ bob measure epc -o my_epc.pdf dev-scores1 eval-scores1
    """
    process = figure.Epc(ctx, scores, True, load.split_files)
    process.run()

@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.output_plot_file_option(default_out='hist.pdf')
@common_options.eval_option()
@common_options.n_bins_option()
@common_options.criterion_option()
@common_options.thresholds_option()
@common_options.const_layout_option()
@common_options.show_dev_option()
@common_options.print_filenames_option()
@common_options.title_option()
@common_options.titles_option()
@common_options.figsize_option()
@common_options.style_option()
@common_options.const_layout_option()
@verbosity_option()
@click.pass_context
def hist(ctx, scores, evaluation, **kwargs):
    """ Plots histograms of positive and negatives along with threshold
    criterion.

    You need to provide one or more development score file(s) for each experiment.
    You can also provide evaluation files along with dev files. If only dev scores
    are provided, you must use flag `--no-evaluation`.

    By default, when eval-scores are given, only eval-scores histograms are 
    displayed with threshold line
    computed from dev-scores. If you want to display dev-scores distributions 
    as well, use ``--show-dev`` option.

    Examples:
        $ bob measure hist dev-scores

        $ bob measure hist dev-scores1 eval-scores1 dev-scores2
        eval-scores2

        $ bob measure hist --criter hter --show-dev dev-scores1 eval-scores1
    """
    process = figure.Hist(ctx, scores, evaluation, load.split_files)
    process.run()

@click.command()
@common_options.scores_argument(nargs=-1)
@common_options.titles_option()
@common_options.sep_dev_eval_option()
@common_options.table_option()
@common_options.eval_option()
@common_options.output_plot_metric_option()
@common_options.output_plot_file_option(default_out='eval_plots.pdf')
@common_options.points_curve_option()
@common_options.semilogx_option(dflt=True)
@common_options.n_bins_option()
@common_options.lines_at_option()
@common_options.const_layout_option()
@common_options.figsize_option()
@common_options.style_option()
@common_options.const_layout_option()
@verbosity_option()
@click.pass_context
def evaluate(ctx, scores, evaluation, **kwargs):
    '''Runs error analysis on score sets

    \b
    1. Computes the threshold using either EER or min. HTER criteria on
       development set scores
    2. Applies the above threshold on evaluation set scores to compute the HTER, if a
       eval-score set is provided
    3. Reports error rates on the console
    4. Plots ROC, EPC, DET curves and score distributions to a multi-page PDF
       file


    You need to provide 2 score files for each biometric system in this order:

    \b
    * development scores
    * evaluation scores

    Examples:
        $ bob measure evaluate dev-scores

        $ bob measure evaluate scores-dev1 scores-eval1 scores-dev2
        scores-eval2

        $ bob measure evaluate /path/to/sys-{1,2,3}/scores-{dev,eval}

        $ bob measure evaluate -l metrics.txt -o my_plots.pdf dev-scores eval-scores
    '''
    # first time erase if existing file
    ctx.meta['open_mode'] = 'w'
    click.echo("Computing metrics with EER...")
    ctx.meta['criter'] = 'eer'  # no criterion passed to evaluate
    ctx.invoke(metrics, scores=scores, evaluation=evaluation)
    # second time, appends the content
    ctx.meta['open_mode'] = 'a'
    click.echo("Computing metrics with HTER...")
    ctx.meta['criter'] = 'hter'  # no criterion passed in evaluate
    ctx.invoke(metrics, scores=scores, evaluation=evaluation)
    if 'log' in ctx.meta:
        click.echo("[metrics] => %s" % ctx.meta['log'])

    # avoid closing pdf file before all figures are plotted
    ctx.meta['closef'] = False
    if evaluation:
        click.echo("Starting evaluate with dev and eval scores...")
    else:
        click.echo("Starting evaluate with dev scores only...")
    click.echo("Computing ROC...")
    # set axes limits for ROC
    ctx.forward(roc) # use class defaults plot settings
    click.echo("Computing DET...")
    ctx.forward(det) # use class defaults plot settings
    if evaluation:
        click.echo("Computing EPC...")
        ctx.forward(epc) # use class defaults plot settings
    # the last one closes the file
    ctx.meta['closef'] = True
    click.echo("Computing score histograms...")
    ctx.meta['criter'] = 'eer'  # no criterion passed in evaluate
    ctx.forward(hist)
    click.echo("Evaluate successfully completed!")
    click.echo("[plots] => %s" % (ctx.meta['output']))
