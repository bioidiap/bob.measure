'''Stores click common options for plots'''

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
                ctx.meta['scores'] = value
                if test_mandatory or ctx.meta['test']:
                    if (length % 2) != 0:
                        pref = 'T' if test_mandatory else ('When `--test` flag'
                                                           ' is on t')
                        raise click.BadParameter(
                            '%sest-score(s) must '
                            'be provided along with dev-score(s)' % pref, ctx=ctx)
                    else:
                        ctx.meta['dev-scores'] = [value[i] for i in
                                                  range(length) if not i % 2]
                        ctx.meta['test-scores'] = [value[i] for i in
                                                   range(length) if i % 2]
                        ctx.meta['n_sys'] = len(ctx.meta['test-scores'])
            return value
        return click.argument('scores', callback=callback, **kwargs)(func)
    return custom_scores_argument

def test_option(**kwargs):
    '''Get option flag to say if test-scores are provided'''
    def custom_test_option(func):
        def callback(ctx, param, value):
            ctx.meta['test'] = value
            return value
        return click.option(
            '-t', '--test/--no-test', default=False,
            help='If set, test scores must be provided',
            callback=callback, is_eager=True ,**kwargs)(func)
    return custom_test_option

def n_sys_option(**kwargs):
    '''Get the number of systems to be processed'''
    def custom_n_sys_option(func):
        def callback(ctx, param, value):
            ctx.meta['n_sys'] = value
            return value
        return click.option(
            '--n-sys', type=INT, default=1, show_default=True,
            help='The number of systems to be processed',
            callback=callback, is_eager=True ,**kwargs)(func)
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
            ctx.meta['n'] = value
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
            if value < 2:
                raise click.BadParameter(
                    'Number of bins must be greater than 1'
                    , ctx=ctx
                )
            ctx.meta['n_bins'] = value
            return value
        return click.option(
            '-b', '--nbins', type=INT, default=20, show_default=True,
            help='The number of bins in the histogram(s)',
            callback=callback, **kwargs)(func)
    return custom_n_bins_option

@click.option('-n', '--points', type=INT, default=100, show_default=True,
              help='Number of points to use in the curves')
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
                value = 'fancy_grid'
            ctx.meta['tablefmt'] = value
            return value
        return click.option(
            '--tablefmt', type=click.STRING, default=None,
            show_default=True, help='Format for table display: `plain`, '
            '`simple`, `grid`, (default) `fancy_grid`, `pipe`, `orgtbl`, '
            '`jira`, `presto`, `psql`, `rst`, `mediawiki`, `moinmoin`, '
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
            ctx.meta['open-mode'] = value
            return value
        return click.option(
            '-om', '--open-mode', default='w',
            help='File open mode',
            callback=callback, **kwargs)(func)
    return custom_open_file_mode_option

def criterion_option(**kwargs):
    '''Get option flag to tell which criteriom is used (default:eer)'''
    def custom_criterion_option(func):
        def callback(ctx, param, value):
            list_accepted_crit = ['eer', 'hter']
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

def label_option(name_option='x-label', **kwargs):
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
            ctx.meta['titles'] = value if value is None else \
                    value.split(',')
            return value
        return click.option(
            '--titles', help='The title for each system comma separated. '
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
