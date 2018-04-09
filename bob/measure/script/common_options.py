'''Stores click common options for plots'''

import math
import pkg_resources  # to make sure bob gets imported properly
import logging
import click
from click.types import INT, FLOAT, Choice, File
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from bob.extension.scripts.click_helper import verbosity_option

logger = logging.getLogger(__name__)

def scores_argument(test_mandatory=False, **kwargs):
    '''Get the argument for scores, and add `dev-scores` and `test-scores` in
    the context if `--test` flag is on (default `--no-test`).'''
    def custom_scores_argument(func):
        def callback(ctx, param, value):
            length = len(value)
            if length < 1:
                raise click.BadParameter('No scores provided', ctx=ctx)
            else:
                div = 1
                ctx.meta['scores'] = value
                if test_mandatory or ctx.meta['test']:
                    div = 2
                    if (length % 2) != 0:
                        pref = 'T' if test_mandatory else ('When `--test` flag'
                                                           ' is on t')
                        raise click.BadParameter(
                            '%sest-score(s) must '
                            'be provided along with dev-score(s)' % pref, ctx=ctx
                        )
                    else:
                        ctx.meta['dev-scores'] = [value[i] for i in
                                                  range(length) if not i % 2]
                        ctx.meta['test-scores'] = [value[i] for i in
                                                   range(length) if i % 2]
                        ctx.meta['n_sys'] = len(ctx.meta['test-scores'])
                if 'titles' in ctx.meta and \
                   len(ctx.meta['titles']) != len(value) / div:
                    raise click.BadParameter(
                        '#titles not equal to #sytems', ctx=ctx
                    )
            return value
        return click.argument(
            'scores', type=click.Path(exists=True),
            callback=callback, **kwargs
        )(func)
    return custom_scores_argument

def bool_option(name, short_name, desc, dflt=False, **kwargs):
    '''Generic provider for boolean options
    Parameters
    ----------

    name: str
        name of the option
    short_name: str
        short name for the option
    desc: str
        short description for the option
    dflt: bool or None
        Default value
     '''
    def custom_bool_option(func):
        def callback(ctx, param, value):
            ctx.meta[name.replace('-', '_')] = value
            return value
        return click.option(
            '-' + short_name, '--%s/--no-%s' % (name, name), default=dflt,
            help=desc,
            show_default=True, callback=callback, is_eager=True, **kwargs)(func)
    return custom_bool_option

def test_option(**kwargs):
    '''Get option flag to say if test-scores are provided'''
    return bool_option('test', 't', 'If set, test scores must be provided')

def sep_dev_test_option(dflt=True, **kwargs):
    '''Get option flag to say if dev and test plots should be in different
    plots'''
    return bool_option(
        'split', 's','If set, test and dev curve in different plots', dflt
    )

def cmc_option(**kwargs):
    '''Get option flag to say if cmc scores'''
    return bool_option('cmc', 'C', 'If set, CMC score files are provided')

def semilogx_option(dflt= False, **kwargs):
    '''Option to use semilog X-axis'''
    return bool_option('semilogx', 'G', 'If set, use semilog on X axis', dflt)

def list_float_option(name, short_name, desc, nitems=None, dflt=None, **kwargs):
    '''Get option to get a list of float f
    Parameters
    ----------

    name: str
        name of the option
    short_name: str
        short name for the option
    desc: str
        short description for the option
    nitems: :obj:`int`
        if given, the parsed list must contains this number of items
    dflt: :any:`list`
        List of default  values for axes.
    '''
    def custom_list_float_option(func):
        def callback(ctx, param, value):
            if value is not None:
                tmp = value.split(',')
                if nitems is not None and len(tmp) != nitems:
                    raise click.BadParameter(
                        '%s Must provide %d axis limits' % (name, nitems)
                    )
                try:
                    value = [float(i) for i in tmp]
                except:
                    raise click.BadParameter('Inputs of %s be floats' % name)
                if None in value:
                    value = None
                elif dflt is not None and None not in dflt and len(dflt) == nitems:
                    value = dflt if not all(
                        isinstance(x, float) for x in dflt
                    ) else None
            ctx.meta[name.replace('-', '_')] = value
            return value
        return click.option(
            '-'+short_name, '--'+name, default=None, show_default=True,
            help=desc, callback=callback, **kwargs)(func)
    return custom_list_float_option

def axes_val_option(dflt=None, **kwargs):
    return list_float_option(
        name='axlim', short_name='L',
        desc='min/max axes values separated by commas (min_x, max_x, min_y, max_y)',
        nitems=4, dflt=dflt, **kwargs
    )

def thresholds_option(**kwargs):
    return list_float_option(
        name='thres', short_name='T',
        desc='Given threshold for metrics computations',
        nitems=None, dflt=None, **kwargs
    )

def axis_fontsize_option(dflt=8, **kwargs):
    '''Get option for axis font size'''
    def custom_axis_fontsize_option(func):
        def callback(ctx, param, value):
            value = abs(value)
            ctx.meta['fontsize'] = value
            return value
        return click.option(
            '-F', '--fontsize', type=click.INT, default=dflt, show_default=True,
            help='Axis fontsize',
            callback=callback, **kwargs)(func)
    return custom_axis_fontsize_option

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

def fmr_line_at_option(**kwargs):
    '''Get option to draw const fmr line'''
    def custom_fmr_line_at_option(func):
        def callback(ctx, param, value):
            if value is not None:
                value = min(max(0.0, value), 1.0)
            ctx.meta['fmr_at'] = value
            return value
        return click.option(
            '-L', '--fmr-at', type=float, default=None, show_default=True,
            help='If given, draw a veritcal line for constant FMR on ROC plots',
            callback=callback, **kwargs)(func)
    return custom_fmr_line_at_option

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

def n_sys_option(**kwargs):
    '''Get the number of systems to be processed'''
    def custom_n_sys_option(func):
        def callback(ctx, param, value):
            ctx.meta['n_sys'] = value
            return value
        return click.option(
            '--n-sys', type=INT, default=1, show_default=True,
            help='The number of systems to be processed',
            callback=callback, is_eager=True , **kwargs)(func)
    return custom_n_sys_option

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
            if value is not None:
                ctx.meta['tablefmt'] = value
            elif 'log' in ctx.meta and ctx.meta['log'] is not None:
                value = 'latex'
            else:
                value = 'rst'
            ctx.meta['tablefmt'] = value
            return value
        return click.option(
            '--tablefmt', type=click.STRING, default=None,
            show_default=True, help='Format for table display: `plain`, '
            '`simple`, `grid`, `fancy_grid`, `pipe`, `orgtbl`, '
            '`jira`, `presto`, `psql`, (default) `rst`, `mediawiki`, `moinmoin`, '
            '`youtrack`, `html`, (default with `--log`)`latex`, '
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
            logger.debug("Plots will be output in %s", value)
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
                logger.debug("Metrics will be output in %s", value)
            ctx.meta['log'] = value
            return value
        return click.option(
            '-l', '--log', default=None, type=click.STRING,
            help='If provided, computed numbers are written to '
              'this file instead of the standard output.',
            callback=callback, **kwargs)(func)
    return custom_output_plot_file_option

def open_file_mode_option(**kwargs):
    '''Get the top option for matplotlib'''
    def custom_open_file_mode_option(func):
        def callback(ctx, param, value):
            if value not in ['w', 'a', 'w+', 'a+']:
                raise click.BadParameter('Incorrect open file mode')
            ctx.meta['open_mode'] = value
            return value
        return click.option(
            '-om', '--open-mode', default='w',
            help='File open mode',
            callback=callback, **kwargs)(func)
    return custom_open_file_mode_option

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

def rank_option(**kwargs):
    '''Get option for rank parameter'''
    def custom_rank_option(func):
        def callback(ctx, param, value):
            value = 1 if value < 0 else value
            ctx.meta['rank'] = value
            return value
        return click.option(
            '--rank', type=click.INT, default=1,
            help='Given threshold for metrics computations',
            callback=callback, show_default=True,**kwargs)(func)
    return custom_rank_option

def label_option(name_option='x_label', **kwargs):
    '''Get labels options based on the given name.

    Parameters:
    ----------
    name_option: str, optional
        Name of the label option (e.g. x-lable, y1-label)
    '''
    def custom_label_option(func):
        def callback(ctx, param, value):
            ''' Get and save labels list in the context list '''
            ctx.meta[name_option] = value if value is None else \
                    [int(i) for i in value.split(',')]
            return value
        return click.option(
            '--' + name_option,
            help='The id of figures which should have x_label separated by '
            'comma. For example ``--%s 1,2,4``.' % name_option,
            callback=callback, **kwargs)(func)
    return custom_label_option

def figsize_option(**kwargs):
    '''Get option for matplotlib figsize'''
    def custom_figsize_option(func):
        def callback(ctx, param, value):
            ctx.meta['figsize'] = value if value is None else \
                    [float(x) for x in value.split(',')]
            plt.figure(figsize=ctx.meta['figsize'])
            return value
        return click.option(
            '--figsize', help='If given, will run \
            ``plt.figure(figsize=figsize)(f)``. Example: --fig-size 4,6',
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

def top_option(**kwargs):
    '''Get the top option for matplotlib'''
    def custom_top_option(func):
        def callback(ctx, param, value):
            ctx.meta['top'] = value
            return value
        return click.option(
            '--top', type=FLOAT,
            help='To give to ``plt.subplots_adjust(top=top)``. If given, first'
            ' plt.tight_layout is called. If you want to tight_layout to be '
            'called, then you need to provide this option.',
            callback=callback, **kwargs)(func)
    return custom_top_option

def titles_option(**kwargs):
    '''Get the titles otpion for the different systems'''
    def custom_titles_option(func):
        def callback(ctx, param, value):
            if value is not None:
                value = value.split(',')
            ctx.meta['titles'] = value
            return value
        return click.option(
            '--titles', type=click.STRING, default=None,
            help='The title for each system comma separated. '
            'Example: --titles ISV,CNN',
            callback=callback, **kwargs)(func)
    return custom_titles_option

def style_option(**kwargs):
    '''Get option for matplotlib style'''
    def custom_style_option(func):
        def callback(ctx, param, value):
            ctx.meta['style'] = value
            plt.style.use(value)
            return value
        return click.option(
            '--style', multiple=True, type=click.types.Choice(plt.style.available),
            help='The matplotlib style to use for plotting. You can provide '
            'multiple styles by repeating this option',
            callback=callback, **kwargs)(func)
    return custom_style_option
