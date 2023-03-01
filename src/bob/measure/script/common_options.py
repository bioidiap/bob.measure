"""Stores click common options for plots"""

import functools
import logging

import click
import matplotlib.pyplot as plt
import tabulate

from clapper.click import verbosity_option
from click.types import FLOAT, INT
from matplotlib.backends.backend_pdf import PdfPages

LOGGER = logging.getLogger(__name__)


def bool_option(name, short_name, desc, dflt=False, **kwargs):
    """Generic provider for boolean options

    Parameters
    ----------
    name : str
        name of the option
    short_name : str
        short name for the option
    desc : str
        short description for the option
    dflt : bool or None
        Default value
    **kwargs
        All kwargs are passed to click.option.

    Returns
    -------
    ``callable``
        A decorator to be used for adding this option.
    """

    def custom_bool_option(func):
        def callback(ctx, param, value):
            ctx.meta[name.replace("-", "_")] = value
            return value

        return click.option(
            "-%s/-n%s" % (short_name, short_name),
            "--%s/--no-%s" % (name, name),
            default=dflt,
            help=desc,
            show_default=True,
            callback=callback,
            is_eager=True,
            **kwargs,
        )(func)

    return custom_bool_option


def list_float_option(name, short_name, desc, nitems=None, dflt=None, **kwargs):
    """Get option to get a list of float f

    Parameters
    ----------
    name : str
        name of the option
    short_name : str
        short name for the option
    desc : str
        short description for the option
    nitems : obj:`int`, optional
        If given, the parsed list must contains this number of items.
    dflt : :any:`list`, optional
        List of default  values for axes.
    **kwargs
        All kwargs are passed to click.option.

    Returns
    -------
    ``callable``
        A decorator to be used for adding this option.
    """

    def custom_list_float_option(func):
        def callback(ctx, param, value):
            if value is None or not value.replace(" ", ""):
                value = None
            elif value is not None:
                tmp = value.split(",")
                if nitems is not None and len(tmp) != nitems:
                    raise click.BadParameter(
                        "%s Must provide %d axis limits" % (name, nitems)
                    )
                try:
                    value = [float(i) for i in tmp]
                except Exception:
                    raise click.BadParameter("Inputs of %s be floats" % name)
            ctx.meta[name.replace("-", "_")] = value
            return value

        return click.option(
            "-" + short_name,
            "--" + name,
            default=dflt,
            show_default=True,
            help=desc + " Provide just a space (' ') to cancel default values.",
            callback=callback,
            **kwargs,
        )(func)

    return custom_list_float_option


def open_file_mode_option(**kwargs):
    """Get open mode file option

    Parameters
    ----------
    **kwargs
        All kwargs are passed to click.option.

    Returns
    -------
    ``callable``
        A decorator to be used for adding this option.
    """

    def custom_open_file_mode_option(func):
        def callback(ctx, param, value):
            if value not in ["w", "a", "w+", "a+"]:
                raise click.BadParameter("Incorrect open file mode")
            ctx.meta["open_mode"] = value
            return value

        return click.option(
            "-om",
            "--open-mode",
            default="w",
            help="File open mode",
            callback=callback,
            **kwargs,
        )(func)

    return custom_open_file_mode_option


def scores_argument(min_arg=1, force_eval=False, **kwargs):
    """Get the argument for scores, and add `dev-scores` and `eval-scores` in
    the context when `--eval` flag is on (default)

    Parameters
    ----------
    min_arg : int
        the minimum number of file needed to evaluate a system. For example,
        vulnerability analysis needs licit and spoof and therefore min_arg = 2

    Returns
    -------
     callable
      A decorator to be used for adding score arguments for click commands
    """

    def custom_scores_argument(func):
        def callback(ctx, param, value):
            min_a = min_arg or 1
            mutli = 1
            error = ""
            if (
                "evaluation" in ctx.meta and ctx.meta["evaluation"]
            ) or force_eval:
                mutli += 1
                error += "- %d evaluation file(s) \n" % min_a
            if "train" in ctx.meta and ctx.meta["train"]:
                mutli += 1
                error += "- %d training file(s) \n" % min_a
            # add more test here if other inputs are needed

            min_a *= mutli
            ctx.meta["min_arg"] = min_a
            if len(value) < 1 or len(value) % ctx.meta["min_arg"] != 0:
                raise click.BadParameter(
                    "The number of provided scores must be > 0 and a multiple of %d "
                    "because the following files are required:\n"
                    "- %d development file(s)\n" % (min_a, min_arg or 1)
                    + error,
                    ctx=ctx,
                )
            ctx.meta["scores"] = value
            return value

        return click.argument(
            "scores", type=click.Path(exists=True), callback=callback, **kwargs
        )(func)

    return custom_scores_argument


def alpha_option(dflt=1, **kwargs):
    """An alpha option for plots"""

    def custom_eval_option(func):
        def callback(ctx, param, value):
            ctx.meta["alpha"] = value
            return value

        return click.option(
            "-a",
            "--alpha",
            default=dflt,
            type=click.FLOAT,
            show_default=True,
            help="Adjusts transparency of plots.",
            callback=callback,
            **kwargs,
        )(func)

    return custom_eval_option


def no_legend_option(dflt=True, **kwargs):
    """Get option flag to say if legend should be displayed or not"""
    return bool_option(
        "disp-legend", "dl", "If set, no legend will be printed.", dflt=dflt
    )


def eval_option(**kwargs):
    """Get option flag to say if eval-scores are provided"""

    def custom_eval_option(func):
        def callback(ctx, param, value):
            ctx.meta["evaluation"] = value
            return value

        return click.option(
            "-e",
            "--eval",
            "evaluation",
            is_flag=True,
            default=False,
            show_default=True,
            help="If set, evaluation scores must be provided",
            callback=callback,
            **kwargs,
        )(func)

    return custom_eval_option


def hide_dev_option(dflt=False, **kwargs):
    """Get option flag to say if dev plot should be hidden"""

    def custom_hide_dev_option(func):
        def callback(ctx, param, value):
            ctx.meta["hide_dev"] = value
            return value

        return click.option(
            "--hide-dev",
            is_flag=True,
            default=dflt,
            show_default=True,
            help="If set, hide dev related plots",
            callback=callback,
            **kwargs,
        )(func)

    return custom_hide_dev_option


def sep_dev_eval_option(dflt=True, **kwargs):
    """Get option flag to say if dev and eval plots should be in different
    plots"""
    return bool_option(
        "split",
        "s",
        "If set, evaluation and dev curve in different plots",
        dflt,
    )


def linestyles_option(dflt=False, **kwargs):
    """Get option flag to turn on/off linestyles"""
    return bool_option(
        "line-styles",
        "S",
        "If given, applies a different line style to each line.",
        dflt,
        **kwargs,
    )


def cmc_option(**kwargs):
    """Get option flag to say if cmc scores"""
    return bool_option(
        "cmc", "C", "If set, CMC score files are provided", **kwargs
    )


def semilogx_option(dflt=False, **kwargs):
    """Option to use semilog X-axis"""
    return bool_option(
        "semilogx", "G", "If set, use semilog on X axis", dflt, **kwargs
    )


def tpr_option(dflt=False, **kwargs):
    """Option to use TPR (true positive rate) on y-axis"""
    return bool_option(
        "tpr",
        "tpr",
        "If set, use TPR (also called 1-FNR, 1-FNMR, or 1-BPCER) on Y axis",
        dflt,
        **kwargs,
    )


def print_filenames_option(dflt=True, **kwargs):
    """Option to tell if filenames should be in the title"""
    return bool_option(
        "show-fn", "P", "If set, show filenames in title", dflt, **kwargs
    )


def const_layout_option(dflt=True, **kwargs):
    """Option to set matplotlib constrained_layout"""

    def custom_layout_option(func):
        def callback(ctx, param, value):
            ctx.meta["clayout"] = value
            plt.rcParams["figure.constrained_layout.use"] = value
            return value

        return click.option(
            "-Y/-nY",
            "--clayout/--no-clayout",
            default=dflt,
            show_default=True,
            help="(De)Activate constrained layout",
            callback=callback,
            **kwargs,
        )(func)

    return custom_layout_option


def axes_val_option(dflt=None, **kwargs):
    """Option for setting min/max values on axes"""
    return list_float_option(
        name="axlim",
        short_name="L",
        desc="min/max axes values separated by commas (e.g. ``--axlim "
        " 0.1,100,0.1,100``)",
        nitems=4,
        dflt=dflt,
        **kwargs,
    )


def thresholds_option(**kwargs):
    """Option to give a list of thresholds"""
    return list_float_option(
        name="thres",
        short_name="T",
        desc="Given threshold for metrics computations, e.g. "
        "0.005,0.001,0.056",
        nitems=None,
        dflt=None,
        **kwargs,
    )


def lines_at_option(dflt="1e-3", **kwargs):
    """Get option to draw const far line"""
    return list_float_option(
        name="lines-at",
        short_name="la",
        desc="If given, draw vertical lines at the given axis positions. "
        "Your values must be separated with a comma (,) without space. "
        "This option works in ROC and DET curves.",
        nitems=None,
        dflt=dflt,
        **kwargs,
    )


def x_rotation_option(dflt=0, **kwargs):
    """Get option for rotartion of the x axis lables"""

    def custom_x_rotation_option(func):
        def callback(ctx, param, value):
            value = abs(value)
            ctx.meta["x_rotation"] = value
            return value

        return click.option(
            "-r",
            "--x-rotation",
            type=click.INT,
            default=dflt,
            show_default=True,
            help="X axis labels ration",
            callback=callback,
            **kwargs,
        )(func)

    return custom_x_rotation_option


def legend_ncols_option(dflt=3, **kwargs):
    """Get option for number of columns for legends"""

    def custom_legend_ncols_option(func):
        def callback(ctx, param, value):
            value = abs(value)
            ctx.meta["legends_ncol"] = value
            return value

        return click.option(
            "-lc",
            "--legends-ncol",
            type=click.INT,
            default=dflt,
            show_default=True,
            help="The number of columns of the legend layout.",
            callback=callback,
            **kwargs,
        )(func)

    return custom_legend_ncols_option


def subplot_option(dflt=111, **kwargs):
    """Get option to set subplots"""

    def custom_subplot_option(func):
        def callback(ctx, param, value):
            value = abs(value)
            nrows = value // 10
            nrows, ncols = divmod(nrows, 10)
            ctx.meta["n_col"] = ncols
            ctx.meta["n_row"] = nrows
            return value

        return click.option(
            "-sp",
            "--subplot",
            type=click.INT,
            default=dflt,
            show_default=True,
            help="The order of subplots.",
            callback=callback,
            **kwargs,
        )(func)

    return custom_subplot_option


def cost_option(**kwargs):
    """Get option to get cost for FPR"""

    def custom_cost_option(func):
        def callback(ctx, param, value):
            if value < 0 or value > 1:
                raise click.BadParameter("Cost for FPR must be betwen 0 and 1")
            ctx.meta["cost"] = value
            return value

        return click.option(
            "-C",
            "--cost",
            type=float,
            default=0.99,
            show_default=True,
            help="Cost for FPR in minDCF",
            callback=callback,
            **kwargs,
        )(func)

    return custom_cost_option


def points_curve_option(**kwargs):
    """Get the number of points use to draw curves"""

    def custom_points_curve_option(func):
        def callback(ctx, param, value):
            if value < 2:
                raise click.BadParameter(
                    "Number of points to draw curves must be greater than 1",
                    ctx=ctx,
                )
            ctx.meta["points"] = value
            return value

        return click.option(
            "-n",
            "--points",
            type=INT,
            default=2000,
            show_default=True,
            help="The number of points use to draw curves in plots",
            callback=callback,
            **kwargs,
        )(func)

    return custom_points_curve_option


def n_bins_option(**kwargs):
    """Get the number of bins in the histograms"""
    possible_strings = [
        "auto",
        "fd",
        "doane",
        "scott",
        "rice",
        "sturges",
        "sqrt",
    ]

    def custom_n_bins_option(func):
        def callback(ctx, param, value):
            if value is None:
                value = ["doane"]
            else:
                tmp = value.split(",")
                try:
                    value = [
                        int(i) if i not in possible_strings else i for i in tmp
                    ]
                except Exception:
                    raise click.BadParameter("Incorrect number of bins inputs")
            ctx.meta["n_bins"] = value
            return value

        return click.option(
            "-b",
            "--nbins",
            type=click.STRING,
            default="doane",
            help="The number of bins for the different quantities to plot, "
            "seperated by commas. For example, if you plot histograms "
            "of negative and positive scores "
            ", input something like `100,doane`. All the "
            "possible bin options can be found in https://docs.scipy.org/doc/"
            "numpy/reference/generated/numpy.histogram.html. Be aware that "
            "for some corner cases, the option `auto` and `fd` can lead to "
            "MemoryError.",
            callback=callback,
            **kwargs,
        )(func)

    return custom_n_bins_option


def table_option(dflt="rst", **kwargs):
    """Get table option for tabulate package
    More informnations: https://pypi.org/project/tabulate/
    """

    def custom_table_option(func):
        def callback(ctx, param, value):
            ctx.meta["tablefmt"] = value
            return value

        return click.option(
            "--tablefmt",
            type=click.Choice(tabulate.tabulate_formats),
            default=dflt,
            show_default=True,
            help="Format of printed tables.",
            callback=callback,
            **kwargs,
        )(func)

    return custom_table_option


def output_plot_file_option(default_out="plots.pdf", **kwargs):
    """Get options for output file for plots"""

    def custom_output_plot_file_option(func):
        def callback(ctx, param, value):
            """Save ouput file  and associated pdf in context list,
            print the path of the file in the log"""
            ctx.meta["output"] = value
            ctx.meta["PdfPages"] = PdfPages(value)
            LOGGER.debug("Plots will be output in %s", value)
            return value

        return click.option(
            "-o",
            "--output",
            default=default_out,
            show_default=True,
            help="The file to save the plots in.",
            callback=callback,
            **kwargs,
        )(func)

    return custom_output_plot_file_option


def output_log_metric_option(**kwargs):
    """Get options for output file for metrics"""

    def custom_output_log_file_option(func):
        def callback(ctx, param, value):
            if value is not None:
                LOGGER.debug("Metrics will be output in %s", value)
            ctx.meta["log"] = value
            return value

        return click.option(
            "-l",
            "--log",
            default=None,
            type=click.STRING,
            help="If provided, computed numbers are written to "
            "this file instead of the standard output.",
            callback=callback,
            **kwargs,
        )(func)

    return custom_output_log_file_option


def no_line_option(**kwargs):
    """Get option flag to say if no line should be displayed"""

    def custom_no_line_option(func):
        def callback(ctx, param, value):
            ctx.meta["no_line"] = value
            return value

        return click.option(
            "--no-line",
            is_flag=True,
            default=False,
            show_default=True,
            help="If set does not display vertical lines",
            callback=callback,
            **kwargs,
        )(func)

    return custom_no_line_option


def criterion_option(
    lcriteria=["eer", "min-hter", "far"], check=True, **kwargs
):
    """Get option flag to tell which criteriom is used (default:eer)

    Parameters
    ----------
    lcriteria : :any:`list`
        List of possible criteria
    """

    def custom_criterion_option(func):
        list_accepted_crit = (
            lcriteria if lcriteria is not None else ["eer", "min-hter", "far"]
        )

        def callback(ctx, param, value):
            if value not in list_accepted_crit and check:
                raise click.BadParameter(
                    "Incorrect value for `--criterion`. "
                    "Must be one of [`%s`]" % "`, `".join(list_accepted_crit)
                )
            ctx.meta["criterion"] = value
            return value

        return click.option(
            "-c",
            "--criterion",
            default="eer",
            help="Criterion to compute plots and "
            "metrics: %s)" % ", ".join(list_accepted_crit),
            callback=callback,
            is_eager=True,
            show_default=True,
            **kwargs,
        )(func)

    return custom_criterion_option


def decimal_option(dflt=1, short="-d", **kwargs):
    """Get option to get decimal value"""

    def custom_decimal_option(func):
        def callback(ctx, param, value):
            ctx.meta["decimal"] = value
            return value

        return click.option(
            short,
            "--decimal",
            type=click.INT,
            default=dflt,
            help="Number of decimals to be printed.",
            callback=callback,
            show_default=True,
            **kwargs,
        )(func)

    return custom_decimal_option


def far_option(far_name="FAR", **kwargs):
    """Get option to get far value"""

    def custom_far_option(func):
        def callback(ctx, param, value):
            if value is not None and (value > 1 or value < 0):
                raise click.BadParameter(
                    "{} value should be between 0 and 1".format(far_name)
                )
            ctx.meta["far_value"] = value
            return value

        return click.option(
            "-f",
            "--{}-value".format(far_name.lower()),
            "far_value",
            type=click.FLOAT,
            default=None,
            help="The {} value for which to compute threshold. This option "
            "must be used alongside `--criterion far`.".format(far_name),
            callback=callback,
            show_default=True,
            **kwargs,
        )(func)

    return custom_far_option


def min_far_option(far_name="FAR", dflt=1e-4, **kwargs):
    """Get option to get min far value"""

    def custom_min_far_option(func):
        def callback(ctx, param, value):
            if value is not None and (value > 1 or value < 0):
                raise click.BadParameter(
                    "{} value should be between 0 and 1".format(far_name)
                )
            ctx.meta["min_far_value"] = value
            return value

        return click.option(
            "-M",
            "--min-{}-value".format(far_name.lower()),
            "min_far_value",
            type=click.FLOAT,
            default=dflt,
            help="Select the minimum {} value used in ROC and DET plots; "
            "should be a power of 10.".format(far_name),
            callback=callback,
            show_default=True,
            **kwargs,
        )(func)

    return custom_min_far_option


def figsize_option(dflt="4,3", **kwargs):
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
            ctx.meta["figsize"] = (
                value if value is None else [float(x) for x in value.split(",")]
            )
            if value is not None:
                plt.rcParams["figure.figsize"] = ctx.meta["figsize"]
            return value

        return click.option(
            "--figsize",
            default=dflt,
            show_default=True,
            help="If given, will run "
            "``plt.rcParams['figure.figsize']=figsize)``. "
            "Example: --figsize 4,6",
            callback=callback,
            **kwargs,
        )(func)

    return custom_figsize_option


def legend_loc_option(dflt="best", **kwargs):
    """Get the legend location of the plot"""

    def custom_legend_loc_option(func):
        def callback(ctx, param, value):
            ctx.meta["legend_loc"] = value.replace("-", " ") if value else value
            return value

        return click.option(
            "-ll",
            "--legend-loc",
            default=dflt,
            show_default=True,
            type=click.Choice(
                [
                    "best",
                    "upper-right",
                    "upper-left",
                    "lower-left",
                    "lower-right",
                    "right",
                    "center-left",
                    "center-right",
                    "lower-center",
                    "upper-center",
                    "center",
                ]
            ),
            help="The legend location code",
            callback=callback,
            **kwargs,
        )(func)

    return custom_legend_loc_option


def line_width_option(**kwargs):
    """Get line width option for the plots"""

    def custom_line_width_option(func):
        def callback(ctx, param, value):
            ctx.meta["line_width"] = value
            return value

        return click.option(
            "--line-width",
            type=FLOAT,
            help="The line width of plots",
            callback=callback,
            **kwargs,
        )(func)

    return custom_line_width_option


def marker_style_option(**kwargs):
    """Get marker style otpion for the plots"""

    def custom_marker_style_option(func):
        def callback(ctx, param, value):
            ctx.meta["marker_style"] = value
            return value

        return click.option(
            "--marker-style",
            type=FLOAT,
            help="The marker style of the plots",
            callback=callback,
            **kwargs,
        )(func)

    return custom_marker_style_option


def legends_option(**kwargs):
    """Get the legends option for the different systems"""

    def custom_legends_option(func):
        def callback(ctx, param, value):
            if value is not None:
                value = value.split(",")
            ctx.meta["legends"] = value
            return value

        return click.option(
            "-lg",
            "--legends",
            type=click.STRING,
            default=None,
            help="The legend for each system comma separated. "
            "Example: --legends ISV,CNN",
            callback=callback,
            **kwargs,
        )(func)

    return custom_legends_option


def title_option(**kwargs):
    """Get the title option for the different systems"""

    def custom_title_option(func):
        def callback(ctx, param, value):
            ctx.meta["title"] = value
            return value

        return click.option(
            "-t",
            "--title",
            type=click.STRING,
            default=None,
            help="The title of the plots. Provide just a space (-t ' ') to "
            "remove the titles from figures.",
            callback=callback,
            **kwargs,
        )(func)

    return custom_title_option


def titles_option(**kwargs):
    """Get the titles option for the different plots"""

    def custom_title_option(func):
        def callback(ctx, param, value):
            if value is not None:
                value = value.split(",")
            ctx.meta["titles"] = value or []
            return value or []

        return click.option(
            "-ts",
            "--titles",
            type=click.STRING,
            default=None,
            help="The titles of the plots seperated by commas. "
            'For example, if the figure has two plots, "MyTitleA,MyTitleB" '
            "is a possible input."
            " Provide just a space (-ts ' ') to "
            "remove the titles from figures.",
            callback=callback,
            **kwargs,
        )(func)

    return custom_title_option


def x_label_option(dflt=None, **kwargs):
    """Get the label option for X axis"""

    def custom_x_label_option(func):
        def callback(ctx, param, value):
            ctx.meta["x_label"] = value
            return value

        return click.option(
            "-xl",
            "--x-label",
            type=click.STRING,
            default=dflt,
            show_default=True,
            help="Label for x-axis",
            callback=callback,
            **kwargs,
        )(func)

    return custom_x_label_option


def y_label_option(dflt=None, **kwargs):
    """Get the label option for Y axis"""

    def custom_y_label_option(func):
        def callback(ctx, param, value):
            ctx.meta["y_label"] = value
            return value

        return click.option(
            "-yl",
            "--y-label",
            type=click.STRING,
            default=dflt,
            help="Label for y-axis",
            callback=callback,
            **kwargs,
        )(func)

    return custom_y_label_option


def style_option(**kwargs):
    """Get option for matplotlib style"""

    def custom_style_option(func):
        def callback(ctx, param, value):
            ctx.meta["style"] = value
            plt.style.use(value)
            return value

        return click.option(
            "--style",
            multiple=True,
            type=click.types.Choice(sorted(plt.style.available)),
            help="The matplotlib style to use for plotting. You can provide "
            "multiple styles by repeating this option",
            callback=callback,
            **kwargs,
        )(func)

    return custom_style_option


def metrics_command(
    docstring,
    criteria=("eer", "min-hter", "far"),
    far_name="FAR",
    check_criteria=True,
    **kwarg,
):
    def custom_metrics_command(func):
        func.__doc__ = docstring

        @click.command(**kwarg)
        @scores_argument(nargs=-1)
        @eval_option()
        @table_option()
        @output_log_metric_option()
        @criterion_option(criteria, check=check_criteria)
        @thresholds_option()
        @far_option(far_name=far_name)
        @legends_option()
        @open_file_mode_option()
        @verbosity_option(LOGGER)
        @click.pass_context
        @decimal_option()
        @functools.wraps(func)
        def wrapper(*args, **kwds):
            return func(*args, **kwds)

        return wrapper

    return custom_metrics_command


METRICS_HELP = """Prints a table that contains {names} for a given
    threshold criterion ({criteria}).
    {hter_note}

    You need to provide one or more development score file(s) for each
    experiment. You can also provide evaluation files along with dev files. If
    evaluation scores are provided, you must use flag `--eval`.

    {score_format}

    Resulting table format can be changed using the `--tablefmt`.

    Examples:

        $ {command} -v scores-dev

        $ {command} -e -l results.txt sys1/scores-{{dev,eval}}

        $ {command} -e {{sys1,sys2}}/scores-{{dev,eval}}

    """


def roc_command(docstring, far_name="FAR"):
    def custom_roc_command(func):
        func.__doc__ = docstring

        @click.command()
        @scores_argument(nargs=-1)
        @titles_option()
        @legends_option()
        @no_legend_option()
        @legend_loc_option(dflt=None)
        @sep_dev_eval_option()
        @output_plot_file_option(default_out="roc.pdf")
        @eval_option()
        @tpr_option(True)
        @semilogx_option(True)
        @lines_at_option()
        @axes_val_option()
        @min_far_option(far_name=far_name)
        @x_rotation_option()
        @x_label_option()
        @y_label_option()
        @points_curve_option()
        @const_layout_option()
        @figsize_option()
        @style_option()
        @linestyles_option()
        @alpha_option()
        @verbosity_option(LOGGER)
        @click.pass_context
        @functools.wraps(func)
        def wrapper(*args, **kwds):
            return func(*args, **kwds)

        return wrapper

    return custom_roc_command


ROC_HELP = """Plot ROC (receiver operating characteristic) curve.
    The plot will represent the false match rate on the horizontal axis and the
    false non match rate on the vertical axis.  The values for the axis will be
    computed using :py:func:`bob.measure.roc`.

    You need to provide one or more development score file(s) for each
    experiment. You can also provide evaluation files along with dev files. If
    evaluation scores are provided, you must use flag `--eval`.

    {score_format}

    Examples:

        $ {command} -v scores-dev

        $ {command} -e -v sys1/scores-{{dev,eval}}

        $ {command} -e -v -o my_roc.pdf {{sys1,sys2}}/scores-{{dev,eval}}
    """


def det_command(docstring, far_name="FAR"):
    def custom_det_command(func):
        func.__doc__ = docstring

        @click.command()
        @scores_argument(nargs=-1)
        @output_plot_file_option(default_out="det.pdf")
        @titles_option()
        @legends_option()
        @no_legend_option()
        @legend_loc_option(dflt="upper-right")
        @sep_dev_eval_option()
        @eval_option()
        @axes_val_option(dflt="0.01,95,0.01,95")
        @min_far_option(far_name=far_name)
        @x_rotation_option(dflt=45)
        @x_label_option()
        @y_label_option()
        @points_curve_option()
        @lines_at_option()
        @const_layout_option()
        @figsize_option()
        @style_option()
        @linestyles_option()
        @alpha_option()
        @verbosity_option(LOGGER)
        @click.pass_context
        @functools.wraps(func)
        def wrapper(*args, **kwds):
            return func(*args, **kwds)

        return wrapper

    return custom_det_command


DET_HELP = """Plot DET (detection error trade-off) curve.
    modified ROC curve which plots error rates on both axes
    (false positives on the x-axis and false negatives on the y-axis).

    You need to provide one or more development score file(s) for each
    experiment. You can also provide evaluation files along with dev files. If
    evaluation scores are provided, you must use flag `--eval`.

    {score_format}

    Examples:

        $ {command} -v scores-dev

        $ {command} -e -v sys1/scores-{{dev,eval}}

        $ {command} -e -v -o my_det.pdf {{sys1,sys2}}/scores-{{dev,eval}}
    """


def epc_command(docstring):
    def custom_epc_command(func):
        func.__doc__ = docstring

        @click.command()
        @scores_argument(min_arg=1, force_eval=True, nargs=-1)
        @output_plot_file_option(default_out="epc.pdf")
        @titles_option()
        @legends_option()
        @no_legend_option()
        @legend_loc_option(dflt="upper-center")
        @points_curve_option()
        @const_layout_option()
        @x_label_option()
        @y_label_option()
        @figsize_option()
        @style_option()
        @linestyles_option()
        @alpha_option()
        @verbosity_option(LOGGER)
        @click.pass_context
        @functools.wraps(func)
        def wrapper(*args, **kwds):
            return func(*args, **kwds)

        return wrapper

    return custom_epc_command


EPC_HELP = """Plot EPC (expected performance curve).
    plots the error rate on the eval set depending on a threshold selected
    a-priori on the development set and accounts for varying relative cost
    in [0; 1] of FPR and FNR when calculating the threshold.

    You need to provide one or more development score and eval file(s)
    for each experiment.

    {score_format}

    Examples:

        $ {command} -v scores-{{dev,eval}}

        $ {command} -v -o my_epc.pdf {{sys1,sys2}}/scores-{{dev,eval}}
    """


def hist_command(docstring, far_name="FAR"):
    def custom_hist_command(func):
        func.__doc__ = docstring

        @click.command()
        @scores_argument(nargs=-1)
        @output_plot_file_option(default_out="hist.pdf")
        @eval_option()
        @hide_dev_option()
        @n_bins_option()
        @titles_option()
        @no_legend_option()
        @legend_ncols_option()
        @criterion_option()
        @far_option(far_name=far_name)
        @no_line_option()
        @thresholds_option()
        @subplot_option()
        @const_layout_option()
        @print_filenames_option()
        @figsize_option(dflt=None)
        @style_option()
        @x_label_option()
        @y_label_option()
        @verbosity_option(LOGGER)
        @click.pass_context
        @functools.wraps(func)
        def wrapper(*args, **kwds):
            return func(*args, **kwds)

        return wrapper

    return custom_hist_command


HIST_HELP = """ Plots histograms of positive and negatives along with threshold
    criterion.

    You need to provide one or more development score file(s) for each
    experiment. You can also provide evaluation files along with dev files. If
    evaluation scores are provided, you must use the `--eval` flag. The
    threshold is always computed from development score files.

    By default, when eval-scores are given, only eval-scores histograms are
    displayed with threshold line computed from dev-scores.

    {score_format}

    Examples:

        $ {command} -v scores-dev

        $ {command} -e -v sys1/scores-{{dev,eval}}

        $ {command} -e -v --criterion min-hter {{sys1,sys2}}/scores-{{dev,eval}}
    """


def evaluate_command(
    docstring, criteria=("eer", "min-hter", "far"), far_name="FAR"
):
    def custom_evaluate_command(func):
        func.__doc__ = docstring

        @click.command()
        @scores_argument(nargs=-1)
        @legends_option()
        @sep_dev_eval_option()
        @table_option()
        @eval_option()
        @criterion_option(criteria)
        @far_option(far_name=far_name)
        @output_log_metric_option()
        @output_plot_file_option(default_out="eval_plots.pdf")
        @lines_at_option()
        @points_curve_option()
        @const_layout_option()
        @figsize_option(dflt=None)
        @style_option()
        @linestyles_option()
        @verbosity_option(LOGGER)
        @click.pass_context
        @functools.wraps(func)
        def wrapper(*args, **kwds):
            return func(*args, **kwds)

        return wrapper

    return custom_evaluate_command


EVALUATE_HELP = """Runs error analysis on score sets.

    \b
    1. Computes the threshold using a criteria (EER by default) on
       development set scores
    2. Applies the above threshold on evaluation set scores to compute the
       HTER if a eval-score (use --eval) set is provided.
    3. Reports error rates on the console or in a log file.
    4. Plots ROC, DET, and EPC curves and score distributions to a multi-page
       PDF file

    You need to provide 1 or 2 score files for each biometric system in this
    order:

    \b
    * development scores
    * evaluation scores

    {score_format}

    Examples:

        $ {command} -v dev-scores

        $ {command} -v /path/to/sys-{{1,2,3}}/scores-dev

        $ {command} -e -v /path/to/sys-{{1,2,3}}/scores-{{dev,eval}}

        $ {command} -v -l metrics.txt -o my_plots.pdf dev-scores

    This command is a combination of metrics, roc, det, epc, and hist commands.
    If you want more flexibility in your plots, please use the individual
    commands.
    """


def evaluate_flow(
    ctx, scores, evaluation, metrics, roc, det, epc, hist, **kwargs
):
    # open_mode is always write in this command.
    ctx.meta["open_mode"] = "w"
    criterion = ctx.meta.get("criterion")
    if criterion is not None:
        click.echo("Computing metrics with %s..." % criterion)
        ctx.invoke(metrics, scores=scores, evaluation=evaluation)
        if ctx.meta.get("log") is not None:
            click.echo("[metrics] => %s" % ctx.meta["log"])

    # avoid closing pdf file before all figures are plotted
    ctx.meta["closef"] = False
    if evaluation:
        click.echo("Starting evaluate with dev and eval scores...")
    else:
        click.echo("Starting evaluate with dev scores only...")
    click.echo("Computing ROC...")
    # set axes limits for ROC
    ctx.forward(roc)  # use class defaults plot settings
    click.echo("Computing DET...")
    ctx.forward(det)  # use class defaults plot settings
    if evaluation:
        click.echo("Computing EPC...")
        ctx.forward(epc)  # use class defaults plot settings
    # the last one closes the file
    ctx.meta["closef"] = True
    click.echo("Computing score histograms...")
    ctx.meta["criterion"] = "eer"  # no criterion passed in evaluate
    ctx.forward(hist)
    click.echo("Evaluate successfully completed!")
    click.echo("[plots] => %s" % (ctx.meta["output"]))


def n_protocols_option(required=True, **kwargs):
    """Get option for number of protocols."""

    def custom_n_protocols_option(func):
        def callback(ctx, param, value):
            value = abs(value)
            ctx.meta["protocols_number"] = value
            return value

        return click.option(
            "-pn",
            "--protocols-number",
            type=click.INT,
            show_default=True,
            required=required,
            help="The number of protocols of cross validation.",
            callback=callback,
            **kwargs,
        )(func)

    return custom_n_protocols_option


def multi_metrics_command(
    docstring, criteria=("eer", "min-hter", "far"), far_name="FAR", **kwargs
):
    def custom_metrics_command(func):
        func.__doc__ = docstring

        @click.command("multi-metrics", **kwargs)
        @scores_argument(nargs=-1)
        @eval_option()
        @n_protocols_option()
        @table_option()
        @output_log_metric_option()
        @criterion_option(criteria)
        @thresholds_option()
        @far_option(far_name=far_name)
        @legends_option()
        @open_file_mode_option()
        @verbosity_option(LOGGER)
        @click.pass_context
        @decimal_option()
        @functools.wraps(func)
        def wrapper(*args, **kwds):
            return func(*args, **kwds)

        return wrapper

    return custom_metrics_command


MULTI_METRICS_HELP = """Multi protocol (cross-validation) metrics.

    Prints a table that contains mean and standard deviation of {names} for a given
    threshold criterion ({criteria}). The metrics are averaged over several protocols.
    The idea is that each protocol corresponds to one fold in your cross-validation.

    You need to provide as many as development score files as the number of
    protocols per system. You can also provide evaluation files along with dev
    files. If evaluation scores are provided, you must use flag `--eval`. The
    number of protocols must be provided using the `--protocols-number` option.

    {score_format}

    Resulting table format can be changed using the `--tablefmt`.

    Examples:

        $ {command} -vv -pn 3 {{p1,p2,p3}}/scores-dev

        $ {command} -vv -pn 3 -e {{p1,p2,p3}}/scores-{{dev,eval}}

        $ {command} -vv -pn 3 -e {{sys1,sys2}}/{{p1,p2,p3}}/scores-{{dev,eval}}
    """
