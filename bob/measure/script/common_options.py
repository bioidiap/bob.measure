'''Stores click common options for plots'''

import logging
import click
from click.types import INT, FLOAT
import matplotlib.pyplot as plt
import tabulate
from matplotlib.backends.backend_pdf import PdfPages
from bob.extension.scripts.click_helper import (bool_option, list_float_option)

LOGGER = logging.getLogger(__name__)


def scores_argument(min_arg=1, force_eval=False, **kwargs):
    """Get the argument for scores, and add `dev-scores` and `eval-scores` in
    the context when `--evaluation` flag is on (default)

    Parameters
    ----------
    min_arg : int
        the minimum number of file needed to evaluate a system. For example,
        PAD functionalities needs licit abd spoof and therefore min_arg = 2

    Returns
    -------
     callable
      A decorator to be used for adding score arguments for click commands
    """
    def custom_scores_argument(func):
        def callback(ctx, param, value):
            min_a = min_arg or 1
            mutli = 1
            error = ''
            if ('evaluation' in ctx.meta and ctx.meta['evaluation']) or force_eval:
                mutli += 1
                error += '- %d evaluation file(s) \n' % min_a
            if 'train' in ctx.meta and ctx.meta['train']:
                mutli += 1
                error += '- %d training file(s) \n' % min_a
            # add more test here if other inputs are needed

            min_a *= mutli
            ctx.meta['min_arg'] = min_a
            if len(value) < 1 or len(value) % ctx.meta['min_arg'] != 0:
                raise click.BadParameter(
                    'The number of provided scores must be > 0 and a multiple of %d '
                    'because the following files are required:\n'
                    '- %d development file(s)\n' % (min_a, min_arg or 1) +
                    error, ctx=ctx
                )
            ctx.meta['scores'] = value
            return value
        return click.argument(
            'scores', type=click.Path(exists=True),
            callback=callback, **kwargs
        )(func)
    return custom_scores_argument


def no_legend_option(dflt=True, **kwargs):
    '''Get option flag to say if legend should be displayed or not'''
    return bool_option(
        'disp-legend', 'dl', 'If set, no legend will be printed.',
        dflt=dflt
    )


def eval_option(**kwargs):
    '''Get option flag to say if eval-scores are provided'''
    return bool_option(
        'evaluation', 'e', 'If set, evaluation scores must be provided',
        dflt=True
    )


def sep_dev_eval_option(dflt=True, **kwargs):
    '''Get option flag to say if dev and eval plots should be in different
    plots'''
    return bool_option(
        'split', 's', 'If set, evaluation and dev curve in different plots',
        dflt
    )


def linestyles_option(dflt=False, **kwargs):
    ''' Get option flag to turn on/off linestyles'''
    return bool_option('line-linestyles', 'S', 'If given, applies a different '
                       'linestyles to each line.', dflt, **kwargs)


def cmc_option(**kwargs):
    '''Get option flag to say if cmc scores'''
    return bool_option('cmc', 'C', 'If set, CMC score files are provided',
                       **kwargs)


def semilogx_option(dflt=False, **kwargs):
    '''Option to use semilog X-axis'''
    return bool_option('semilogx', 'G', 'If set, use semilog on X axis', dflt,
                       **kwargs)


def print_filenames_option(dflt=True, **kwargs):
    '''Option to tell if filenames should be in the title'''
    return bool_option('show-fn', 'P', 'If set, show filenames in title', dflt,
                       **kwargs)


def const_layout_option(dflt=True, **kwargs):
    '''Option to set matplotlib constrained_layout'''
    def custom_layout_option(func):
        def callback(ctx, param, value):
            ctx.meta['clayout'] = value
            plt.rcParams['figure.constrained_layout.use'] = value
            return value
        return click.option(
            '-Y', '--clayout/--no-clayout', default=dflt, show_default=True,
            help='(De)Activate constrained layout',
            callback=callback, **kwargs)(func)
    return custom_layout_option


def axes_val_option(dflt=None, **kwargs):
    ''' Option for setting min/max values on axes '''
    return list_float_option(
        name='axlim', short_name='L',
        desc='min/max axes values separated by commas (e.g. ``--axlim '
        ' 0.1,100,0.1,100``)',
        nitems=4, dflt=dflt, **kwargs
    )


def thresholds_option(**kwargs):
    ''' Option to give a list of thresholds '''
    return list_float_option(
        name='thres', short_name='T',
        desc='Given threshold for metrics computations, e.g. '
        '0.005,0.001,0.056',
        nitems=None, dflt=None, **kwargs
    )


def lines_at_option(**kwargs):
    '''Get option to draw const far line'''
    return list_float_option(
        name='lines-at', short_name='la',
        desc='If given, draw vertical lines at the given axis positions. '
        'You can provide multiple values separated with a comma (,).',
        nitems=None, dflt='1e-3', **kwargs
    )


def x_rotation_option(dflt=0, **kwargs):
    '''Get option for rotartion of the x axis lables'''
    def custom_x_rotation_option(func):
        def callback(ctx, param, value):
            value = abs(value)
            ctx.meta['x_rotation'] = value
            return value
        return click.option(
            '-r', '--x-rotation', type=click.INT, default=dflt,
            show_default=True, help='X axis labels ration',
            callback=callback, **kwargs)(func)
    return custom_x_rotation_option


def legend_ncols_option(dflt=10, **kwargs):
    '''Get option for number of columns for legends'''
    def custom_legend_ncols_option(func):
        def callback(ctx, param, value):
            value = abs(value)
            ctx.meta['legends_ncol'] = value
            return value
        return click.option(
            '-lc', '--legends-ncol', type=click.INT, default=dflt,
            show_default=True,
            help='The number of columns of the legend layout.',
            callback=callback, **kwargs)(func)
    return custom_legend_ncols_option


def subplot_option(dflt=111, **kwargs):
    '''Get option to set subplots'''
    def custom_subplot_option(func):
        def callback(ctx, param, value):
            value = abs(value)
            nrows = value // 10
            nrows, ncols = divmod(nrows, 10)
            ctx.meta['n_col'] = ncols
            ctx.meta['n_row'] = nrows
            return value
        return click.option(
            '-sp', '--subplot', type=click.INT, default=dflt,
            show_default=True, help='The order of subplots.',
            callback=callback, **kwargs)(func)
    return custom_subplot_option


def cost_option(**kwargs):
    '''Get option to get cost for FAR'''
    def custom_cost_option(func):
        def callback(ctx, param, value):
            if value < 0 or value > 1:
                raise click.BadParameter("Cost for FAR must be betwen 0 and 1")
            ctx.meta['cost'] = value
            return value
        return click.option(
            '-C', '--cost', type=float, default=0.99, show_default=True,
            help='Cost for FAR in minDCF',
            callback=callback, **kwargs)(func)
    return custom_cost_option


def points_curve_option(**kwargs):
    '''Get the number of points use to draw curves'''
    def custom_points_curve_option(func):
        def callback(ctx, param, value):
            if value < 2:
                raise click.BadParameter(
                    'Number of points to draw curves must be greater than 1',
                    ctx=ctx
                )
            ctx.meta['points'] = value
            return value
        return click.option(
            '-n', '--points', type=INT, default=100, show_default=True,
            help='The number of points use to draw curves in plots',
            callback=callback, **kwargs)(func)
    return custom_points_curve_option


def n_bins_option(**kwargs):
    '''Get the number of bins in the histograms'''
    def custom_n_bins_option(func):
        def callback(ctx, param, value):
            if value is None:
                value = 'auto'
            else:
                tmp = value.split(',')
                try:
                    value = [int(i) if i != 'auto' else i for i in tmp]
                except Exception:
                    raise click.BadParameter('Incorrect number of bins inputs')
            ctx.meta['n_bins'] = value
            return value
        return click.option(
            '-b', '--nbins', type=click.STRING, default='auto',
            help='The number of bins for the different histograms in the '
            ' figure, seperated by commas. For example, if three histograms '
            'are in the plots, input something like `100,auto,50`',
            callback=callback, **kwargs)(func)
    return custom_n_bins_option


def table_option(**kwargs):
    '''Get table option for tabulate package
    More informnations: https://pypi.org/project/tabulate/
    '''
    def custom_table_option(func):
        def callback(ctx, param, value):
            ctx.meta['tablefmt'] = value
            return value
        return click.option(
            '--tablefmt', type=click.Choice(tabulate.tabulate_formats),
            default='rst', show_default=True, help='Format of printed tables.',
            callback=callback, **kwargs)(func)
    return custom_table_option


def output_plot_file_option(default_out='plots.pdf', **kwargs):
    '''Get options for output file for plots'''
    def custom_output_plot_file_option(func):
        def callback(ctx, param, value):
            ''' Save ouput file  and associated pdf in context list,
            print the path of the file in the log'''
            ctx.meta['output'] = value
            ctx.meta['PdfPages'] = PdfPages(value)
            LOGGER.debug("Plots will be output in %s", value)
            return value
        return click.option(
            '-o', '--output',
            default=default_out, show_default=True,
            help='The file to save the plots in.',
            callback=callback, **kwargs)(func)
    return custom_output_plot_file_option


def output_log_metric_option(**kwargs):
    '''Get options for output file for metrics'''
    def custom_output_log_file_option(func):
        def callback(ctx, param, value):
            if value is not None:
                LOGGER.debug("Metrics will be output in %s", value)
            ctx.meta['log'] = value
            return value
        return click.option(
            '-l', '--log', default=None, type=click.STRING,
            help='If provided, computed numbers are written to '
            'this file instead of the standard output.',
            callback=callback, **kwargs)(func)
    return custom_output_log_file_option


def criterion_option(lcriteria=['eer', 'min-hter', 'far'], **kwargs):
    """Get option flag to tell which criteriom is used (default:eer)

    Parameters
    ----------
    lcriteria : :any:`list`
        List of possible criteria
    """
    def custom_criterion_option(func):
        def callback(ctx, param, value):
            list_accepted_crit = lcriteria if lcriteria is not None else \
                ['eer', 'min-hter', 'far']
            if value not in list_accepted_crit:
                raise click.BadParameter('Incorrect value for `--criterion`. '
                                         'Must be one of [`%s`]' %
                                         '`, `'.join(list_accepted_crit))
            ctx.meta['criterion'] = value
            return value
        return click.option(
            '-c', '--criterion', default='eer',
            help='Criterion to compute plots and '
            'metrics: `eer`, `min-hter` or `far`',
            callback=callback, is_eager=True, **kwargs)(func)
    return custom_criterion_option


def far_option(**kwargs):
    '''Get option to get far value'''
    def custom_far_option(func):
        def callback(ctx, param, value):
            if value is not None and (value > 1 or value < 0):
                raise click.BadParameter("FAR value should be between 0 and 1")
            ctx.meta['far_value'] = value
            return value
        return click.option(
            '-f', '--far-value', type=click.FLOAT, default=None,
            help='The FAR value for which to compute metrics',
            callback=callback, show_default=True, **kwargs)(func)
    return custom_far_option


def min_far_option(dflt=1e-4, **kwargs):
    '''Get option to get min far value'''
    def custom_min_far_option(func):
        def callback(ctx, param, value):
            if value is not None and (value > 1 or value < 0):
                raise click.BadParameter("FAR value should be between 0 and 1")
            ctx.meta['min_far_value'] = value
            return value
        return click.option(
            '-M', '--min-far-value', type=click.FLOAT, default=dflt,
            help='Select the minimum FAR value used in ROC and DET plots; '
            'should be a power of 10.',
            callback=callback, show_default=True, **kwargs)(func)
    return custom_min_far_option


def figsize_option(dflt='4,3', **kwargs):
    """Get option for matplotlib figsize

    Parameters
    ----------
    dflt : str
        matplotlib default figsize for the command. must be a a list of int
        separated by commas.

    Returns
    -------
    callable
        A decorator to be used for adding score arguments for click commands
    """
    def custom_figsize_option(func):
        def callback(ctx, param, value):
            ctx.meta['figsize'] = value if value is None else \
                    [float(x) for x in value.split(',')]
            if value is not None:
                plt.rcParams['figure.figsize'] = ctx.meta['figsize']
            return value
        return click.option(
            '--figsize', default=dflt, show_default=True,
            help='If given, will run '
            '``plt.rcParams[\'figure.figsize\']=figsize)``. '
            'Example: --fig-size 4,6',
            callback=callback, **kwargs)(func)
    return custom_figsize_option


def legend_loc_option(dflt=0, **kwargs):
    '''Get the legend location of the plot'''
    def custom_legend_loc_option(func):
        def callback(ctx, param, value):
            ctx.meta['legend_loc'] = value
            return value
        return click.option(
            '-lc', '--legend-loc', default=dflt, show_default=True,
            type=INT, help='The legend location code',
            callback=callback, **kwargs)(func)
    return custom_legend_loc_option


def line_width_option(**kwargs):
    '''Get line width option for the plots'''
    def custom_line_width_option(func):
        def callback(ctx, param, value):
            ctx.meta['line_width'] = value
            return value
        return click.option(
            '--line-width',
            type=FLOAT, help='The line width of plots',
            callback=callback, **kwargs)(func)
    return custom_line_width_option


def marker_style_option(**kwargs):
    '''Get marker style otpion for the plots'''
    def custom_marker_style_option(func):
        def callback(ctx, param, value):
            ctx.meta['marker_style'] = value
            return value
        return click.option(
            '--marker-style',
            type=FLOAT, help='The marker style of the plots',
            callback=callback, **kwargs)(func)
    return custom_marker_style_option


def legends_option(**kwargs):
    '''Get the legends option for the different systems'''
    def custom_legends_option(func):
        def callback(ctx, param, value):
            if value is not None:
                value = value.split(',')
            ctx.meta['legends'] = value
            return value
        return click.option(
            '-lg', '--legends', type=click.STRING, default=None,
            help='The title for each system comma separated. '
            'Example: --legends ISV,CNN',
            callback=callback, **kwargs)(func)
    return custom_legends_option


def title_option(**kwargs):
    '''Get the title option for the different systems'''
    def custom_title_option(func):
        def callback(ctx, param, value):
            ctx.meta['title'] = value
            return value
        return click.option(
            '-t', '--title', type=click.STRING, default=None,
            help="The title of the plots. Provide just a space (-t ' ') to "
            "remove the titles from figures.",
            callback=callback, **kwargs)(func)
    return custom_title_option


def x_label_option(dflt=None, **kwargs):
    '''Get the label option for X axis '''
    def custom_x_label_option(func):
        def callback(ctx, param, value):
            ctx.meta['x_label'] = value
            return value
        return click.option(
            '-xl', '--x-lable', type=click.STRING, default=dflt,
            show_default=True, help='Label for x-axis',
            callback=callback, **kwargs)(func)
    return custom_x_label_option


def y_label_option(dflt=None, **kwargs):
    '''Get the label option for Y axis '''
    def custom_y_label_option(func):
        def callback(ctx, param, value):
            ctx.meta['y_label'] = value
            return value
        return click.option(
            '-yl', '--y-lable', type=click.STRING, default=dflt,
            help='Label for y-axis',
            callback=callback, **kwargs)(func)
    return custom_y_label_option


def style_option(**kwargs):
    '''Get option for matplotlib style'''
    def custom_style_option(func):
        def callback(ctx, param, value):
            ctx.meta['style'] = value
            plt.style.use(value)
            return value
        return click.option(
            '--style', multiple=True,
            type=click.types.Choice(sorted(plt.style.available)),
            help='The matplotlib style to use for plotting. You can provide '
            'multiple styles by repeating this option',
            callback=callback, **kwargs)(func)
    return custom_style_option
