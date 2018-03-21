'''Stores click common options for plots'''

import logging
import click
from click.types import INT, FLOAT, Choice, File
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from bob.extension.scripts.click_helper import verbosity_option

logger = logging.getLogger(__name__)

def plot_options(f):
    # more import options go down the list here.
    f = click.pass_context(f)
    f = verbosity_option()(f)
    f = click.option(
        '--style', multiple=True, type=click.types.Choice(plt.style.available),
        help='The matplotlib style to use for plotting. You can provide '
        'multiple styles by repeating this option')(f)
    f = click.option(
        '--titles', help='The title for each system comma separated. '
        'Example: --titles ISV,CNN')(f)
    f = click.option(
        '--top', type=FLOAT,
        help='To give to ``plt.subplots_adjust(top=top)``. If given, first '
        'plt.tight_layout is called. If you want to tight_layout to be called,'
        ' then you need to provide this option.')(f)
    f = click.option(
        '--legend-ncol', default=3, show_default=True,
        type=INT,
        help='The number of columns of the legend layout.')(f)
    f = click.option(
        '--figsize', help='If given, will run '
        '``plt.figure(figsize=figsize)(f)``. Example: --fig-size 4,6')(f)
    # f = click.option(
    #     '--y2-label',
    #     help='The id of figures which should have y2_label separated by '
    #     'comma. For example ``--y2-label 1,2,4``.')(f)
    f = click.option(
        '--y1-label',
        help='The id of figures which should have y1_label separated by '
        'comma. For example ``--y1-label 1,2,4``.')(f)
    f = click.option(
        '--x-label',
        help='The id of figures which should have x_label separated by '
        'comma. For example ``--x-label 1,2,4``.')(f)
    f = click.option(
        '--subplot', type=INT, default=111,
        show_default=True, help='The order of subplots.')(f)
    f = click.option(
        '-o', '--output', type=File(mode='wb'),
        default='plots.pdf', show_default=True,
        help='The file to save the plots in.')(f)
    return f

def normalize_options(ctx, n_systems, output, subplot, style, x_label,
                      y1_label, figsize, legend_ncol, top, titles,
                      y2_label=None):
    if style:
        plt.style.use(style)

    ctx.meta['output'] = output
    ctx.meta['PdfPages'] = PdfPages(output)

    ctx.meta['x_label'] = x_label if x_label is None else \
        [int(x) for x in x_label.split(',')]
    ctx.meta['y1_label'] = y1_label if y1_label is None else \
        [int(x) for x in y1_label.split(',')]
    ctx.meta['y2_label'] = y2_label if y2_label is None else \
        [int(x) for x in y2_label.split(',')]

    ctx.meta['subplot'] = subplot
    nrows = subplot // 10
    nrows, ncols = divmod(nrows, 10)
    logger.debug('Got %d, %d for nrows and ncols', nrows, ncols)
    ctx.meta['nrows_ncols'] = nrows, ncols

    ctx.meta['figsize'] = figsize if figsize is None else \
        [float(x) for x in figsize.split(',')]
    plt.figure(figsize=ctx.meta['figsize'])

    ctx.meta['legend_ncol'] = legend_ncol
    ctx.meta['top'] = top

    ctx.meta['titles'] = titles if titles is None else titles.split(',')
    nrows, ncols = ctx.meta['nrows_ncols']
    if nrows * ncols < n_systems:
        logger.error("The number of subplots is smaller than the number of "
                     "systems. I will plot one system a column. Use --subplot "
                     "to remove this error.")
        nrows, ncols = 1, n_systems

    ctx.meta['nrows'], ctx.meta['ncols'] = nrows, ncols

    ctx.meta['titles'] = ctx.meta['titles'] or [None] * n_systems

    # Try to automatically figure out where to place labels
    # x_label should be True if row == -1
    # y1_label should be True if col == 0
    # y2_label should be True if col == -1
    ctx.meta['x_label'] = ctx.meta['x_label'] or \
        [x for x in range(1, n_systems + 1)
         if ((x - 1) // ncols) == (nrows - 1)]
    ctx.meta['y1_label'] = ctx.meta['y1_label'] or \
        [x for x in range(1, n_systems + 1)
         if ((x - 1) % ncols) == 0]
    ctx.meta['y2_label'] = ctx.meta.get('y2_label', None) or \
        [x for x in range(1, n_systems + 1)
         if ((x - 1) % ncols) == (ncols - 1)]

    return ctx

