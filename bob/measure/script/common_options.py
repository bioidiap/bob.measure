'''Stores click common options for plots'''

import logging
import click
from click.types import INT, FLOAT
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from bob.extension.scripts.click_helper import (bool_option, list_float_option)

LOGGER = logging.getLogger(__name__)

def scores_argument(eval_mandatory=False, min_len=1, **kwargs):
    """Get the argument for scores, and add `dev-scores` and `eval-scores` in
    the context when `--evaluation` flag is on (default)

    Parameters
    ----------
    eval_mandatory :
        If evaluation files are mandatory
    min_len :
        The min lenght of inputs files that are needed. If eval_mandatory is
        True, this quantity is multiplied by 2.

    Returns
    -------
     callable
      A decorator to be used for adding score arguments for click commands
    """
    def custom_scores_argument(func):
        def callback(ctx, param, value):
            length = len(value)
            min_arg = min_len or 1
            ctx.meta['min_arg'] = min_arg
            if length < min_arg:
                raise click.BadParameter(
                    'You must provide at least %d score files' % min_arg,
                    ctx=ctx
                )
            else:
                ctx.meta['scores'] = value
                step = 1
                if eval_mandatory or ctx.meta['evaluation']:
                    step = 2
                    if (length % (min_arg * 2)) != 0:
                        pref = 'T' if eval_mandatory else \
                                ('When `--evaluation` flag is on t')
                        raise click.BadParameter(
                            '%sest-score(s) must '
                            'be provided along with dev-score(s). '
                            'You must provide at least %d score files.' \
                            % (pref, min_arg * 2), ctx=ctx
                        )
                for arg in range(min_arg):
                    ctx.meta['dev_scores_%d' % arg] = [
                        value[i] for i in range(arg * step, length,
                                                min_arg * step)
                    ]
                    if step > 1:
                        ctx.meta['eval_scores_%d' % arg] = [
                            value[i] for i in range((arg * step + 1),
                                                    length, min_arg * step)
                        ]
                ctx.meta['n_sys'] = len(ctx.meta['dev_scores_0'])
                if 'titles' in ctx.meta and \
                   len(ctx.meta['titles']) != ctx.meta['n_sys']:
                    raise click.BadParameter(
                        '#titles not equal to #sytems', ctx=ctx
                    )
            return value
        return click.argument(
            'scores', type=click.Path(exists=True),
            callback=callback, **kwargs
        )(func)
    return custom_scores_argument

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
        'split', 's','If set, evaluation and dev curve in different plots',
        dflt
    )

def cmc_option(**kwargs):
    '''Get option flag to say if cmc scores'''
    return bool_option('cmc', 'C', 'If set, CMC score files are provided')

def semilogx_option(dflt=False, **kwargs):
    '''Option to use semilog X-axis'''
    return bool_option('semilogx', 'G', 'If set, use semilog on X axis', dflt)

def show_dev_option(dflt=False, **kwargs):
    '''Option to tell if should show dev histo'''
    return bool_option('show-dev', 'D', 'If set, show dev histograms', dflt)

def print_filenames_option(dflt=True, **kwargs):
    '''Option to tell if filenames should be in the title'''
    return bool_option('show-fn', 'P', 'If set, show filenames in title', dflt)

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
        desc='If given, draw veritcal lines at the given axis positions',
        nitems=None, dflt=None, **kwargs
    )

def x_rotation_option(dflt=0, **kwargs):
    '''Get option for rotartion of the x axis lables'''
    def custom_x_rotation_option(func):
        def callback(ctx, param, value):
            value = abs(value)
            ctx.meta['x_rotation'] = value
            return value
        return click.option(
            '-r', '--x-rotation', type=click.INT, default=dflt, show_default=True,
            help='X axis labels ration',
            callback=callback, **kwargs)(func)
    return custom_x_rotation_option

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
                    'Number of points to draw curves must be greater than 1'
                    , ctx=ctx
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
            elif value < 2:
                raise click.BadParameter(
                    'Number of bins must be greater than 1'
                    , ctx=ctx
                )
            ctx.meta['n_bins'] = value
            return value
        return click.option(
            '-b', '--nbins', type=INT, default=None,
            help='The number of bins in the histogram(s). Default: `auto`',
            callback=callback, **kwargs)(func)
    return custom_n_bins_option

def table_option(**kwargs):
    '''Get table option for tabulate package
    More informnations: https://pypi.python.org/pypi/tabulate
    '''
    def custom_table_option(func):
        def callback(ctx, param, value):
            ctx.meta['tablefmt'] = value
            return value
        return click.option(
            '--tablefmt', type=click.STRING, default='rst',
            show_default=True, help='Format for table display: `plain`, '
            '`simple`, `grid`, `fancy_grid`, `pipe`, `orgtbl`, '
            '`jira`, `presto`, `psql`, `rst`, `mediawiki`, `moinmoin`, '
            '`youtrack`, `html`, `latex`, '
            '`latex_raw`, `latex_booktabs`, `textile`',
            callback=callback,**kwargs)(func)
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

def output_plot_metric_option(**kwargs):
    '''Get options for output file for metrics'''
    def custom_output_plot_file_option(func):
        def callback(ctx, param, value):
            ''' Save ouput file  and associated pdf in context list,
            print the path of the file in the log'''
            if value is not None:
                LOGGER.debug("Metrics will be output in %s", value)
            ctx.meta['log'] = value
            return value
        return click.option(
            '-l', '--log', default=None, type=click.STRING,
            help='If provided, computed numbers are written to '
              'this file instead of the standard output.',
            callback=callback, **kwargs)(func)
    return custom_output_plot_file_option

def criterion_option(lcriteria=['eer', 'hter', 'far'], **kwargs):
    """Get option flag to tell which criteriom is used (default:eer)

    Parameters
    ----------
    lcriteria : :any:`list`
        List of possible criteria
    """
    def custom_criterion_option(func):
        def callback(ctx, param, value):
            list_accepted_crit = lcriteria if lcriteria is not None else \
                    ['eer', 'hter', 'far']
            if value not in list_accepted_crit:
                raise click.BadParameter('Incorrect value for `--criter`. '
                                         'Must be one of [`%s`]' %
                                         '`, `'.join(list_accepted_crit))
            ctx.meta['criter'] = value
            return value
        return click.option(
            '--criter', default='eer', help='Criterion to compute plots and '
            'metrics: `eer` (default), `hter`',
            callback=callback, is_eager=True ,**kwargs)(func)
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
            callback=callback, show_default=True,**kwargs)(func)
    return custom_far_option

def figsize_option(**kwargs):
    '''Get option for matplotlib figsize'''
    def custom_figsize_option(func):
        def callback(ctx, param, value):
            ctx.meta['figsize'] = value if value is None else \
                    [float(x) for x in value.split(',')]
            if value is not None:
                plt.rcParams['figure.figsize'] = ctx.meta['figsize']
            return value
        return click.option(
            '--figsize', help='If given, will run '
            '``plt.rcParams[\'figure.figsize\']=figsize)``. Example: --fig-size 4,6',
            callback=callback, **kwargs)(func)
    return custom_figsize_option

def legend_ncols_option(**kwargs):
    '''Get the number of columns to set in the legend of the plot'''
    def custom_legend_ncols_option(func):
        def callback(ctx, param, value):
            ctx.meta['legend_ncol'] = value
            return value
        return click.option(
            '--legend-ncol', default=3, show_default=True,
            type=INT, help='The number of columns of the legend layout.',
            callback=callback, **kwargs)(func)
    return custom_legend_ncols_option

def legend_loc_option(**kwargs):
    '''Get tthe legend location of the plot'''
    def custom_legend_loc_option(func):
        def callback(ctx, param, value):
            ctx.meta['legend_loc'] = value
            return value
        return click.option(
            '--legend-location', default=0, show_default=True,
            type=INT, help='The lengend location code',
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

def titles_option(**kwargs):
    '''Get the titles option for the different systems'''
    def custom_titles_option(func):
        def callback(ctx, param, value):
            if value is not None:
                value = value.split(',')
            ctx.meta['titles'] = value
            return value
        return click.option(
            '-ts', '--titles', type=click.STRING, default=None,
            help='The title for each system comma separated. '
            'Example: --titles ISV,CNN',
            callback=callback, **kwargs)(func)
    return custom_titles_option

def title_option(**kwargs):
    '''Get the title option for the different systems'''
    def custom_title_option(func):
        def callback(ctx, param, value):
            ctx.meta['title'] = value
            return value
        return click.option(
            '-t', '--title', type=click.STRING, default=None,
            help='The title of the plots',
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
            help='Label for x-axis',
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
            '--style', multiple=True, type=click.types.Choice(sorted(plt.style.available)),
            help='The matplotlib style to use for plotting. You can provide '
            'multiple styles by repeating this option',
            callback=callback, **kwargs)(func)
    return custom_style_option
