"""Runs error analysis on score sets, outputs metrics and plots"""

from __future__ import division, print_function

import logging
import math
import sys

from abc import ABCMeta, abstractmethod

import click
import matplotlib
import matplotlib.pyplot as mpl
import numpy

from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages
from tabulate import tabulate

from .. import far_threshold, plot, ppndf, utils

LOGGER = logging.getLogger("bob.measure")


def check_list_value(values, desired_number, name, name2="systems"):
    if values is not None and len(values) != desired_number:
        if len(values) == 1:
            values = values * desired_number
        else:
            raise click.BadParameter(
                "#{} ({}) must be either 1 value or the same as "
                "#{} ({} values)".format(name, values, name2, desired_number)
            )

    return values


class MeasureBase(object):
    """Base class for metrics and plots.
    This abstract class define the framework to plot or compute metrics from a
    list of (positive, negative) scores tuples.

    Attributes
    ----------
    func_load:
        Function that is used to load the input files
    """

    __metaclass__ = ABCMeta  # for python 2.7 compatibility

    def __init__(self, ctx, scores, evaluation, func_load):
        """
        Parameters
        ----------
        ctx : :py:class:`dict`
            Click context dictionary.

        scores : :any:`list`:
            List of input files (e.g. dev-{1, 2, 3}, {dev,eval}-scores1
            {dev,eval}-scores2)
        eval : :py:class:`bool`
            True if eval data are used
        func_load : Function that is used to load the input files
        """
        self._scores = scores
        self._ctx = ctx
        self.func_load = func_load
        self._legends = ctx.meta.get("legends")
        self._eval = evaluation
        self._min_arg = ctx.meta.get("min_arg", 1)
        if len(scores) < 1 or len(scores) % self._min_arg != 0:
            raise click.BadParameter(
                "Number of argument must be a non-zero multiple of %d"
                % self._min_arg
            )
        self.n_systems = int(len(scores) / self._min_arg)
        if self._legends is not None and len(self._legends) < self.n_systems:
            raise click.BadParameter(
                "Number of legends must be >= to the " "number of systems"
            )

    def run(self):
        """Generate outputs (e.g. metrics, files, pdf plots).
        This function calls abstract methods
        :func:`~bob.measure.script.figure.MeasureBase.init_process` (before
        loop), :py:func:`~bob.measure.script.figure.MeasureBase.compute`
        (in the loop iterating through the different
        systems) and :py:func:`~bob.measure.script.figure.MeasureBase.end_process`
        (after the loop).
        """
        # init matplotlib, log files, ...
        self.init_process()
        # iterates through the different systems and feed `compute`
        # with the dev (and eval) scores of each system
        # Note that more than one dev or eval scores score can be passed to
        # each system
        for idx in range(self.n_systems):
            # load scores for each system: get the corresponding arrays and
            # base-name of files
            input_scores, input_names = self._load_files(
                # Scores are given as followed:
                # SysA-dev SysA-eval ... SysA-XX  SysB-dev SysB-eval ... SysB-XX
                # ------------------------------  ------------------------------
                #   First set of `self._min_arg`     Second set of input files
                #     input files starting at               for SysB
                #    index idx * self._min_arg
                self._scores[idx * self._min_arg : (idx + 1) * self._min_arg]
            )
            LOGGER.info("-----Input files for system %d-----", idx + 1)
            for i, name in enumerate(input_names):
                if not self._eval:
                    LOGGER.info("Dev. score %d: %s", i + 1, name)
                else:
                    if i % 2 == 0:
                        LOGGER.info("Dev. score %d: %s", i / 2 + 1, name)
                    else:
                        LOGGER.info("Eval. score %d: %s", i / 2 + 1, name)
            LOGGER.info("----------------------------------")

            self.compute(idx, input_scores, input_names)
        # setup final configuration, plotting properties, ...
        self.end_process()

    # protected functions that need to be overwritten
    def init_process(self):
        """Called in :py:func:`~bob.measure.script.figure.MeasureBase`.run
        before iterating through the different systems.
        Should reimplemented in derived classes"""
        pass

    # Main computations are done here in the subclasses
    @abstractmethod
    def compute(self, idx, input_scores, input_names):
        """Compute metrics or plots from the given scores provided by
        :py:func:`~bob.measure.script.figure.MeasureBase.run`.
        Should reimplemented in derived classes

        Parameters
        ----------
        idx : :obj:`int`
            index of the system
        input_scores: :any:`list`
            list of scores returned by the loading function
        input_names: :any:`list`
            list of base names for the input file of the system
        """
        pass
        # structure of input is (vuln example):
        # if evaluation is provided
        # [ (dev_licit_neg, dev_licit_pos), (eval_licit_neg, eval_licit_pos),
        #   (dev_spoof_neg, dev_licit_pos), (eval_spoof_neg, eval_licit_pos)]
        # and if only dev:
        # [ (dev_licit_neg, dev_licit_pos), (dev_spoof_neg, dev_licit_pos)]

    # Things to do after the main iterative computations are done
    @abstractmethod
    def end_process(self):
        """Called in :py:func:`~bob.measure.script.figure.MeasureBase`.run
        after iterating through the different systems.
        Should reimplemented in derived classes"""
        pass

    # common protected functions

    def _load_files(self, filepaths):
        """Load the input files and return the base names of the files

        Returns
        -------
            scores: :any:`list`:
                A list that contains the output of
                ``func_load`` for the given files
            basenames: :any:`list`:
                A list of the given files
        """
        scores = []
        basenames = []
        for filename in filepaths:
            basenames.append(filename)
            scores.append(self.func_load(filename))
        return scores, basenames


class Metrics(MeasureBase):
    """Compute metrics from score files

    Attributes
    ----------
    log_file: str
        output stream
    """

    def __init__(
        self,
        ctx,
        scores,
        evaluation,
        func_load,
        names=(
            "False Positive Rate",
            "False Negative Rate",
            "Precision",
            "Recall",
            "F1-score",
            "Area Under ROC Curve",
            "Area Under ROC Curve (log scale)",
        ),
    ):
        super(Metrics, self).__init__(ctx, scores, evaluation, func_load)
        self.names = names
        self._tablefmt = ctx.meta.get("tablefmt")
        self._criterion = ctx.meta.get("criterion")
        self._open_mode = ctx.meta.get("open_mode")
        self._thres = ctx.meta.get("thres")
        self._decimal = ctx.meta.get("decimal", 2)
        if self._thres is not None:
            if len(self._thres) == 1:
                self._thres = self._thres * self.n_systems
            elif len(self._thres) != self.n_systems:
                raise click.BadParameter(
                    "#thresholds must be the same as #systems (%d)"
                    % len(self.n_systems)
                )
        self._far = ctx.meta.get("far_value")
        self._log = ctx.meta.get("log")
        self.log_file = sys.stdout
        if self._log is not None:
            self.log_file = open(self._log, self._open_mode)

    def get_thres(self, criterion, dev_neg, dev_pos, far):
        return utils.get_thres(criterion, dev_neg, dev_pos, far)

    def _numbers(self, neg, pos, threshold, fta):
        from .. import f_score, farfrr, precision_recall, roc_auc_score

        # fpr and fnr
        fmr, fnmr = farfrr(neg, pos, threshold)
        hter = (fmr + fnmr) / 2.0
        far = fmr * (1 - fta)
        frr = fta + fnmr * (1 - fta)

        ni = neg.shape[0]  # number of impostors
        fm = int(round(fmr * ni))  # number of false accepts
        nc = pos.shape[0]  # number of clients
        fnm = int(round(fnmr * nc))  # number of false rejects

        # precision and recall
        precision, recall = precision_recall(neg, pos, threshold)

        # f_score
        f1_score = f_score(neg, pos, threshold, 1)

        # AUC ROC
        auc = roc_auc_score(neg, pos)
        auc_log = roc_auc_score(neg, pos, log_scale=True)
        return (
            fta,
            fmr,
            fnmr,
            hter,
            far,
            frr,
            fm,
            ni,
            fnm,
            nc,
            precision,
            recall,
            f1_score,
            auc,
            auc_log,
        )

    def _strings(self, metrics):
        n_dec = ".%df" % self._decimal
        fta_str = "%s%%" % format(100 * metrics[0], n_dec)
        fmr_str = "%s%% (%d/%d)" % (
            format(100 * metrics[1], n_dec),
            metrics[6],
            metrics[7],
        )
        fnmr_str = "%s%% (%d/%d)" % (
            format(100 * metrics[2], n_dec),
            metrics[8],
            metrics[9],
        )
        far_str = "%s%%" % format(100 * metrics[4], n_dec)
        frr_str = "%s%%" % format(100 * metrics[5], n_dec)
        hter_str = "%s%%" % format(100 * metrics[3], n_dec)
        prec_str = "%s" % format(metrics[10], n_dec)
        recall_str = "%s" % format(metrics[11], n_dec)
        f1_str = "%s" % format(metrics[12], n_dec)
        auc_str = "%s" % format(metrics[13], n_dec)
        auc_log_str = "%s" % format(metrics[14], n_dec)

        return (
            fta_str,
            fmr_str,
            fnmr_str,
            far_str,
            frr_str,
            hter_str,
            prec_str,
            recall_str,
            f1_str,
            auc_str,
            auc_log_str,
        )

    def _get_all_metrics(self, idx, input_scores, input_names):
        """Compute all metrics for dev and eval scores"""
        neg_list, pos_list, fta_list = utils.get_fta_list(input_scores)
        dev_neg, dev_pos, dev_fta = neg_list[0], pos_list[0], fta_list[0]
        dev_file = input_names[0]
        if self._eval:
            eval_neg, eval_pos, eval_fta = neg_list[1], pos_list[1], fta_list[1]

        threshold = (
            self.get_thres(self._criterion, dev_neg, dev_pos, self._far)
            if self._thres is None
            else self._thres[idx]
        )

        title = self._legends[idx] if self._legends is not None else None
        if self._thres is None:
            far_str = ""
            if self._criterion == "far" and self._far is not None:
                far_str = str(self._far)
            click.echo(
                "[Min. criterion: %s %s] Threshold on Development set `%s`: %e"
                % (
                    self._criterion.upper(),
                    far_str,
                    title or dev_file,
                    threshold,
                ),
                file=self.log_file,
            )
        else:
            click.echo(
                "[Min. criterion: user provided] Threshold on "
                "Development set `%s`: %e" % (dev_file or title, threshold),
                file=self.log_file,
            )

        res = []
        res.append(
            self._strings(self._numbers(dev_neg, dev_pos, threshold, dev_fta))
        )

        if self._eval:
            # computes statistics for the eval set based on the threshold a
            # priori
            res.append(
                self._strings(
                    self._numbers(eval_neg, eval_pos, threshold, eval_fta)
                )
            )
        else:
            res.append(None)

        return res

    def compute(self, idx, input_scores, input_names):
        """Compute metrics thresholds and tables (FPR, FNR, precision, recall,
        f1_score) for given system inputs"""
        dev_file = input_names[0]
        title = self._legends[idx] if self._legends is not None else None
        all_metrics = self._get_all_metrics(idx, input_scores, input_names)
        fta_dev = float(all_metrics[0][0].replace("%", ""))
        if fta_dev > 0.0:
            LOGGER.warn(
                "NaNs scores (%s) were found in %s amd removed",
                all_metrics[0][0],
                dev_file,
            )
        headers = [" " or title, "Development"]
        rows = [
            [self.names[0], all_metrics[0][1]],
            [self.names[1], all_metrics[0][2]],
            [self.names[2], all_metrics[0][6]],
            [self.names[3], all_metrics[0][7]],
            [self.names[4], all_metrics[0][8]],
            [self.names[5], all_metrics[0][9]],
            [self.names[6], all_metrics[0][10]],
        ]

        if self._eval:
            eval_file = input_names[1]
            fta_eval = float(all_metrics[1][0].replace("%", ""))
            if fta_eval > 0.0:
                LOGGER.warn(
                    "NaNs scores (%s) were found in %s and removed.",
                    all_metrics[1][0],
                    eval_file,
                )
            # computes statistics for the eval set based on the threshold a
            # priori
            headers.append("Evaluation")
            rows[0].append(all_metrics[1][1])
            rows[1].append(all_metrics[1][2])
            rows[2].append(all_metrics[1][6])
            rows[3].append(all_metrics[1][7])
            rows[4].append(all_metrics[1][8])
            rows[5].append(all_metrics[1][9])
            rows[6].append(all_metrics[1][10])

        click.echo(tabulate(rows, headers, self._tablefmt), file=self.log_file)

    def end_process(self):
        """Close log file if needed"""
        if self._log is not None:
            self.log_file.close()


class MultiMetrics(Metrics):
    """Computes average of metrics based on several protocols (cross
    validation)

    Attributes
    ----------
    log_file : str
        output stream
    names : tuple
        List of names for the metrics.
    """

    def __init__(
        self,
        ctx,
        scores,
        evaluation,
        func_load,
        names=(
            "NaNs Rate",
            "False Positive Rate",
            "False Negative Rate",
            "False Accept Rate",
            "False Reject Rate",
            "Half Total Error Rate",
        ),
    ):
        super(MultiMetrics, self).__init__(
            ctx, scores, evaluation, func_load, names=names
        )

        self.headers = ["Methods"] + list(self.names)
        if self._eval:
            self.headers.insert(1, self.names[5] + " (dev)")
        self.rows = []

    def _strings(self, metrics):
        (
            ftam,
            fmrm,
            fnmrm,
            hterm,
            farm,
            frrm,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = metrics.mean(axis=0)
        ftas, fmrs, fnmrs, hters, fars, frrs, _, _, _, _, _, _, _ = metrics.std(
            axis=0
        )
        n_dec = ".%df" % self._decimal
        fta_str = "%s%% (%s%%)" % (
            format(100 * ftam, n_dec),
            format(100 * ftas, n_dec),
        )
        fmr_str = "%s%% (%s%%)" % (
            format(100 * fmrm, n_dec),
            format(100 * fmrs, n_dec),
        )
        fnmr_str = "%s%% (%s%%)" % (
            format(100 * fnmrm, n_dec),
            format(100 * fnmrs, n_dec),
        )
        far_str = "%s%% (%s%%)" % (
            format(100 * farm, n_dec),
            format(100 * fars, n_dec),
        )
        frr_str = "%s%% (%s%%)" % (
            format(100 * frrm, n_dec),
            format(100 * frrs, n_dec),
        )
        hter_str = "%s%% (%s%%)" % (
            format(100 * hterm, n_dec),
            format(100 * hters, n_dec),
        )
        return fta_str, fmr_str, fnmr_str, far_str, frr_str, hter_str

    def compute(self, idx, input_scores, input_names):
        """Computes the average of metrics over several protocols."""
        neg_list, pos_list, fta_list = utils.get_fta_list(input_scores)
        step = 2 if self._eval else 1
        self._dev_metrics = []
        self._thresholds = []
        for i in range(0, len(input_scores), step):
            neg, pos, fta = neg_list[i], pos_list[i], fta_list[i]
            threshold = (
                self.get_thres(self._criterion, neg, pos, self._far)
                if self._thres is None
                else self._thres[idx]
            )
            self._thresholds.append(threshold)
            self._dev_metrics.append(self._numbers(neg, pos, threshold, fta))
        self._dev_metrics = numpy.array(self._dev_metrics)

        if self._eval:
            self._eval_metrics = []
            for i in range(1, len(input_scores), step):
                neg, pos, fta = neg_list[i], pos_list[i], fta_list[i]
                threshold = self._thresholds[i // 2]
                self._eval_metrics.append(
                    self._numbers(neg, pos, threshold, fta)
                )
            self._eval_metrics = numpy.array(self._eval_metrics)

        title = self._legends[idx] if self._legends is not None else None

        fta_str, fmr_str, fnmr_str, far_str, frr_str, hter_str = self._strings(
            self._dev_metrics
        )

        if self._eval:
            self.rows.append([title, hter_str])
        else:
            self.rows.append(
                [title, fta_str, fmr_str, fnmr_str, far_str, frr_str, hter_str]
            )

        if self._eval:
            # computes statistics for the eval set based on the threshold a
            # priori
            (
                fta_str,
                fmr_str,
                fnmr_str,
                far_str,
                frr_str,
                hter_str,
            ) = self._strings(self._eval_metrics)

            self.rows[-1].extend(
                [fta_str, fmr_str, fnmr_str, far_str, frr_str, hter_str]
            )

    def end_process(self):
        click.echo(
            tabulate(self.rows, self.headers, self._tablefmt),
            file=self.log_file,
        )
        super(MultiMetrics, self).end_process()


class PlotBase(MeasureBase):
    """Base class for plots. Regroup several options and code
    shared by the different plots
    """

    def __init__(self, ctx, scores, evaluation, func_load):
        super(PlotBase, self).__init__(ctx, scores, evaluation, func_load)
        self._output = ctx.meta.get("output")
        self._points = ctx.meta.get("points", 2000)
        self._split = ctx.meta.get("split")
        self._axlim = ctx.meta.get("axlim")
        self._alpha = ctx.meta.get("alpha")
        self._disp_legend = ctx.meta.get("disp_legend", True)
        self._legend_loc = ctx.meta.get("legend_loc")
        self._min_dig = None
        if "min_far_value" in ctx.meta:
            self._min_dig = int(math.log10(ctx.meta["min_far_value"]))
        elif self._axlim is not None and self._axlim[0] is not None:
            self._min_dig = int(
                math.log10(self._axlim[0]) if self._axlim[0] != 0 else 0
            )
        self._clayout = ctx.meta.get("clayout")
        self._far_at = ctx.meta.get("lines_at")
        self._trans_far_val = self._far_at
        if self._far_at is not None:
            self._eval_points = {line: [] for line in self._far_at}
            self._lines_val = []
        self._print_fn = ctx.meta.get("show_fn", True)
        self._x_rotation = ctx.meta.get("x_rotation")
        if "style" in ctx.meta:
            mpl.style.use(ctx.meta["style"])
        self._nb_figs = 2 if self._eval and self._split else 1
        self._colors = utils.get_colors(self.n_systems)
        self._line_linestyles = ctx.meta.get("line_styles", False)
        self._linestyles = utils.get_linestyles(
            self.n_systems, self._line_linestyles
        )
        self._titles = ctx.meta.get("titles", []) * 2
        # for compatibility
        self._title = ctx.meta.get("title")
        if not self._titles and self._title is not None:
            self._titles = [self._title] * 2

        self._x_label = ctx.meta.get("x_label")
        self._y_label = ctx.meta.get("y_label")
        self._grid_color = "silver"
        self._pdf_page = None
        self._end_setup_plot = True

    def init_process(self):
        """Open pdf and set axis font size if provided"""
        if not hasattr(matplotlib, "backends"):
            matplotlib.use("pdf")

        self._pdf_page = (
            self._ctx.meta["PdfPages"]
            if "PdfPages" in self._ctx.meta
            else PdfPages(self._output)
        )

        for i in range(self._nb_figs):
            fs = self._ctx.meta.get("figsize")
            fig = mpl.figure(i + 1, figsize=fs)
            fig.set_constrained_layout(self._clayout)
            fig.clear()

    def end_process(self):
        """Set title, legend, axis labels, grid colors, save figures, drow
        lines and close pdf if needed"""
        # draw vertical lines
        if self._far_at is not None:
            for line, line_trans in zip(self._far_at, self._trans_far_val):
                mpl.figure(1)
                mpl.plot(
                    [line_trans, line_trans],
                    [-100.0, 100.0],
                    "--",
                    color="black",
                )
                if self._eval and self._split:
                    mpl.figure(2)
                    x_values = [i for i, _ in self._eval_points[line]]
                    y_values = [j for _, j in self._eval_points[line]]
                    sort_indice = sorted(
                        range(len(x_values)), key=x_values.__getitem__
                    )
                    x_values = [x_values[i] for i in sort_indice]
                    y_values = [y_values[i] for i in sort_indice]
                    mpl.plot(x_values, y_values, "--", color="black")
        # only for plots
        if self._end_setup_plot:
            for i in range(self._nb_figs):
                fig = mpl.figure(i + 1)
                title = "" if not self._titles else self._titles[i]
                mpl.title(title if title.replace(" ", "") else "")
                mpl.xlabel(self._x_label)
                mpl.ylabel(self._y_label)
                mpl.grid(True, color=self._grid_color)
                if self._disp_legend:
                    self.plot_legends()
                self._set_axis()
                mpl.xticks(rotation=self._x_rotation)
                self._pdf_page.savefig(fig)

        # do not want to close PDF when running evaluate
        if "PdfPages" in self._ctx.meta and (
            "closef" not in self._ctx.meta or self._ctx.meta["closef"]
        ):
            self._pdf_page.close()

    def plot_legends(self):
        """Print legend on current plot"""
        if not self._disp_legend:
            return

        lines = []
        labels = []
        for ax in mpl.gcf().get_axes():
            ali, ala = ax.get_legend_handles_labels()
            # avoid duplicates in legend
            for li, la in zip(ali, ala):
                if la not in labels:
                    lines.append(li)
                    labels.append(la)

        # create legend on the top or bottom axis
        leg = mpl.legend(
            lines,
            labels,
            loc=self._legend_loc,
            ncol=1,
        )

        return leg

    # common protected functions

    def _label(self, base, idx):
        if self._legends is not None and len(self._legends) > idx:
            return self._legends[idx]
        if self.n_systems > 1:
            return base + (" %d" % (idx + 1))
        return base

    def _set_axis(self):
        if self._axlim is not None:
            mpl.axis(self._axlim)


class Roc(PlotBase):
    """Handles the plotting of ROC"""

    def __init__(self, ctx, scores, evaluation, func_load):
        super(Roc, self).__init__(ctx, scores, evaluation, func_load)
        self._titles = self._titles or ["ROC dev.", "ROC eval."]
        self._x_label = self._x_label or "FPR"
        self._semilogx = ctx.meta.get("semilogx", True)
        self._tpr = ctx.meta.get("tpr", True)
        dflt_y_label = "TPR" if self._tpr else "FNR"
        self._y_label = self._y_label or dflt_y_label
        best_legend = "lower right" if self._semilogx else "upper right"
        self._legend_loc = self._legend_loc or best_legend
        # custom defaults
        if self._axlim is None:
            self._axlim = [None, None, -0.05, 1.05]
        self._min_dig = -4 if self._min_dig is None else self._min_dig

    def compute(self, idx, input_scores, input_names):
        """Plot ROC for dev and eval data using
        :py:func:`bob.measure.plot.roc`"""
        neg_list, pos_list, _ = utils.get_fta_list(input_scores)
        dev_neg, dev_pos = neg_list[0], pos_list[0]
        dev_file = input_names[0]
        if self._eval:
            eval_neg, eval_pos = neg_list[1], pos_list[1]
            eval_file = input_names[1]

        mpl.figure(1)
        if self._eval:
            LOGGER.info("ROC dev. curve using %s", dev_file)
            plot.roc(
                dev_neg,
                dev_pos,
                npoints=self._points,
                semilogx=self._semilogx,
                tpr=self._tpr,
                min_far=self._min_dig,
                color=self._colors[idx],
                linestyle=self._linestyles[idx],
                label=self._label("dev", idx),
                alpha=self._alpha,
            )
            if self._split:
                mpl.figure(2)

            linestyle = "--" if not self._split else self._linestyles[idx]
            LOGGER.info("ROC eval. curve using %s", eval_file)
            plot.roc(
                eval_neg,
                eval_pos,
                linestyle=linestyle,
                npoints=self._points,
                semilogx=self._semilogx,
                tpr=self._tpr,
                min_far=self._min_dig,
                color=self._colors[idx],
                label=self._label("eval.", idx),
                alpha=self._alpha,
            )
            if self._far_at is not None:
                from .. import fprfnr

                for line in self._far_at:
                    thres_line = far_threshold(dev_neg, dev_pos, line)
                    eval_fmr, eval_fnmr = fprfnr(eval_neg, eval_pos, thres_line)
                    if self._tpr:
                        eval_fnmr = 1 - eval_fnmr
                    mpl.scatter(eval_fmr, eval_fnmr, c=self._colors[idx], s=30)
                    self._eval_points[line].append((eval_fmr, eval_fnmr))
        else:
            LOGGER.info("ROC dev. curve using %s", dev_file)
            plot.roc(
                dev_neg,
                dev_pos,
                npoints=self._points,
                semilogx=self._semilogx,
                tpr=self._tpr,
                min_far=self._min_dig,
                color=self._colors[idx],
                linestyle=self._linestyles[idx],
                label=self._label("dev", idx),
                alpha=self._alpha,
            )


class Det(PlotBase):
    """Handles the plotting of DET"""

    def __init__(self, ctx, scores, evaluation, func_load):
        super(Det, self).__init__(ctx, scores, evaluation, func_load)
        self._titles = self._titles or ["DET dev.", "DET eval."]
        self._x_label = self._x_label or "FPR (%)"
        self._y_label = self._y_label or "FNR (%)"
        self._legend_loc = self._legend_loc or "upper right"
        if self._far_at is not None:
            self._trans_far_val = ppndf(self._far_at)
        # custom defaults here
        if self._x_rotation is None:
            self._x_rotation = 50

        if self._axlim is None:
            self._axlim = [0.01, 99, 0.01, 99]

        if self._min_dig is not None:
            self._axlim[0] = math.pow(10, self._min_dig) * 100

        self._min_dig = -4 if self._min_dig is None else self._min_dig

    def compute(self, idx, input_scores, input_names):
        """Plot DET for dev and eval data using
        :py:func:`bob.measure.plot.det`"""
        neg_list, pos_list, _ = utils.get_fta_list(input_scores)
        dev_neg, dev_pos = neg_list[0], pos_list[0]
        dev_file = input_names[0]
        if self._eval:
            eval_neg, eval_pos = neg_list[1], pos_list[1]
            eval_file = input_names[1]

        mpl.figure(1)
        if self._eval and eval_neg is not None:
            LOGGER.info("DET dev. curve using %s", dev_file)
            plot.det(
                dev_neg,
                dev_pos,
                self._points,
                min_far=self._min_dig,
                color=self._colors[idx],
                linestyle=self._linestyles[idx],
                label=self._label("dev.", idx),
                alpha=self._alpha,
            )
            if self._split:
                mpl.figure(2)
            linestyle = "--" if not self._split else self._linestyles[idx]
            LOGGER.info("DET eval. curve using %s", eval_file)
            plot.det(
                eval_neg,
                eval_pos,
                self._points,
                min_far=self._min_dig,
                color=self._colors[idx],
                linestyle=linestyle,
                label=self._label("eval.", idx),
                alpha=self._alpha,
            )
            if self._far_at is not None:
                from .. import farfrr

                for line in self._far_at:
                    thres_line = far_threshold(dev_neg, dev_pos, line)
                    eval_fmr, eval_fnmr = farfrr(eval_neg, eval_pos, thres_line)
                    eval_fmr, eval_fnmr = ppndf(eval_fmr), ppndf(eval_fnmr)
                    mpl.scatter(eval_fmr, eval_fnmr, c=self._colors[idx], s=30)
                    self._eval_points[line].append((eval_fmr, eval_fnmr))
        else:
            LOGGER.info("DET dev. curve using %s", dev_file)
            plot.det(
                dev_neg,
                dev_pos,
                self._points,
                min_far=self._min_dig,
                color=self._colors[idx],
                linestyle=self._linestyles[idx],
                label=self._label("dev.", idx),
                alpha=self._alpha,
            )

    def _set_axis(self):
        plot.det_axis(self._axlim)


class Epc(PlotBase):
    """Handles the plotting of EPC"""

    def __init__(self, ctx, scores, evaluation, func_load, hter="HTER"):
        super(Epc, self).__init__(ctx, scores, evaluation, func_load)
        if self._min_arg != 2:
            raise click.UsageError("EPC requires dev. and eval. score files")
        self._titles = self._titles or ["EPC"] * 2
        self._x_label = self._x_label or r"$\alpha$"
        self._y_label = self._y_label or hter + " (%)"
        self._legend_loc = self._legend_loc or "upper center"
        self._eval = True  # always eval data with EPC
        self._split = False
        self._nb_figs = 1
        self._far_at = None

    def compute(self, idx, input_scores, input_names):
        """Plot EPC using :py:func:`bob.measure.plot.epc`"""
        neg_list, pos_list, _ = utils.get_fta_list(input_scores)
        dev_neg, dev_pos = neg_list[0], pos_list[0]
        dev_file = input_names[0]
        if self._eval:
            eval_neg, eval_pos = neg_list[1], pos_list[1]
            eval_file = input_names[1]

        LOGGER.info("EPC using %s", dev_file + "_" + eval_file)
        plot.epc(
            dev_neg,
            dev_pos,
            eval_neg,
            eval_pos,
            self._points,
            color=self._colors[idx],
            linestyle=self._linestyles[idx],
            label=self._label("curve", idx),
            alpha=self._alpha,
        )


class GridSubplot(PlotBase):
    """A base class for plots that contain subplots and legends.

    To use this class, use `create_subplot` in `compute` each time you need a
    new axis. and call `finalize_one_page` in `compute` when a page is finished
    rendering.
    """

    def __init__(self, ctx, scores, evaluation, func_load):
        super(GridSubplot, self).__init__(ctx, scores, evaluation, func_load)

        # Check legend
        self._legend_loc = self._legend_loc or "upper center"
        if self._legend_loc == "best":
            self._legend_loc = "upper center"
        if "upper" not in self._legend_loc and "lower" not in self._legend_loc:
            raise ValueError(
                "Only best, upper-*, and lower-* legend locations are supported!"
            )
        self._nlegends = ctx.meta.get("legends_ncol", 3)

        # subplot grid
        self._nrows = ctx.meta.get("n_row", 1)
        self._ncols = ctx.meta.get("n_col", 1)

    def init_process(self):
        super(GridSubplot, self).init_process()
        self._create_grid_spec()

    def _create_grid_spec(self):
        # create a compatible GridSpec
        self._gs = gridspec.GridSpec(
            self._nrows,
            self._ncols,
            figure=mpl.gcf(),
        )

    def create_subplot(self, n, shared_axis=None):
        i, j = numpy.unravel_index(n, (self._nrows, self._ncols))
        axis = mpl.gcf().add_subplot(
            self._gs[i : i + 1, j : j + 1], sharex=shared_axis
        )
        return axis

    def finalize_one_page(self):
        # print legend on the page
        self.plot_legends()
        fig = mpl.gcf()
        axes = fig.get_axes()

        LOGGER.debug("%s contains %d axes:", fig, len(axes))
        for i, ax in enumerate(axes, start=1):
            LOGGER.debug("Axes %d: %s", i, ax)

        self._pdf_page.savefig(bbox_inches="tight")
        mpl.clf()
        mpl.figure()
        self._create_grid_spec()

    def plot_legends(self):
        """Print legend on current page"""
        if not self._disp_legend:
            return

        lines = []
        labels = []
        for ax in mpl.gcf().get_axes():
            ali, ala = ax.get_legend_handles_labels()
            # avoid duplicates in legend
            for li, la in zip(ali, ala):
                if la not in labels:
                    lines.append(li)
                    labels.append(la)

        # create legend on the top or bottom axis
        fig = mpl.gcf()
        if "upper" in self._legend_loc:
            # Set anchor to top of figure
            bbox_to_anchor = (0.0, 1.0, 1.0, 0.0)
            # Legend will be anchored with its bottom side, so switch the loc
            anchored_loc = self._legend_loc.replace("upper", "lower")
        else:
            # Set anchor to bottom of figure
            bbox_to_anchor = (0.0, 0.0, 1.0, 0.0)
            # Legend will be anchored with its top side, so switch the loc
            anchored_loc = self._legend_loc.replace("lower", "upper")
        leg = fig.legend(
            lines,
            labels,
            loc=anchored_loc,
            ncol=self._nlegends,
            bbox_to_anchor=bbox_to_anchor,
        )

        return leg


class Hist(GridSubplot):
    """Functional base class for histograms"""

    def __init__(self, ctx, scores, evaluation, func_load, nhist_per_system=2):
        super(Hist, self).__init__(ctx, scores, evaluation, func_load)
        self._nbins = ctx.meta.get("n_bins", ["doane"])
        self._nhist_per_system = nhist_per_system
        self._nbins = check_list_value(
            self._nbins, nhist_per_system, "n_bins", "histograms"
        )
        self._thres = ctx.meta.get("thres")
        self._thres = check_list_value(
            self._thres, self.n_systems, "thresholds"
        )
        self._criterion = ctx.meta.get("criterion")
        # no vertical (threshold) is displayed
        self._no_line = ctx.meta.get("no_line", False)
        # do not display dev histo
        self._hide_dev = ctx.meta.get("hide_dev", False)
        if self._hide_dev and not self._eval:
            raise click.BadParameter(
                "You can only use --hide-dev along with --eval"
            )
        # dev hist are displayed next to eval hist
        self._nrows *= 1 if self._hide_dev or not self._eval else 2
        self._nlegends = ctx.meta.get("legends_ncol", 3)

        # number of subplot on one page
        self._step_print = int(self._nrows * self._ncols)
        self._title_base = "Scores"
        self._y_label = self._y_label or "Probability density"
        self._x_label = self._x_label or "Score values"
        self._end_setup_plot = False
        # overide _titles of PlotBase
        self._titles = ctx.meta.get("titles", []) * 2

    def compute(self, idx, input_scores, input_names):
        """Draw histograms of negative and positive scores."""
        (
            dev_neg,
            dev_pos,
            eval_neg,
            eval_pos,
            threshold,
        ) = self._get_neg_pos_thres(idx, input_scores, input_names)

        # keep id of the current system
        sys = idx
        # if the id of the current system does not match the id of the plot,
        # change it
        if not self._hide_dev and self._eval:
            row = int(idx / self._ncols) * 2
            col = idx % self._ncols
            idx = col + self._ncols * row

        dev_axis = None

        if not self._hide_dev or not self._eval:
            dev_axis = self._print_subplot(
                idx,
                sys,
                dev_neg,
                dev_pos,
                threshold,
                not self._no_line,
                False,
            )

        if self._eval:
            idx += self._ncols if not self._hide_dev else 0
            self._print_subplot(
                idx,
                sys,
                eval_neg,
                eval_pos,
                threshold,
                not self._no_line,
                True,
                shared_axis=dev_axis,
            )

    def _print_subplot(
        self,
        idx,
        sys,
        neg,
        pos,
        threshold,
        draw_line,
        evaluation,
        shared_axis=None,
    ):
        """print a subplot for the given score and subplot index"""
        n = idx % self._step_print
        col = n % self._ncols
        sub_plot_idx = n + 1
        axis = self.create_subplot(n, shared_axis)
        self._setup_hist(neg, pos)
        if col == 0:
            axis.set_ylabel(self._y_label)
        # systems per page
        sys_per_page = self._step_print / (
            1 if self._hide_dev or not self._eval else 2
        )
        # rest to be printed
        sys_idx = sys % sys_per_page
        rest_print = self.n_systems - int(sys / sys_per_page) * sys_per_page
        # lower histo only
        is_lower = evaluation or not self._eval
        if is_lower and sys_idx + self._ncols >= min(sys_per_page, rest_print):
            axis.set_xlabel(self._x_label)
        dflt_title = "Eval. scores" if evaluation else "Dev. scores"
        if self.n_systems == 1 and (not self._eval or self._hide_dev):
            dflt_title = " "
        add = self.n_systems if is_lower else 0
        axis.set_title(self._get_title(sys + add, dflt_title))
        label = "%s threshold%s" % (
            "" if self._criterion is None else self._criterion.upper(),
            " (dev)" if self._eval else "",
        )
        if draw_line:
            self._lines(threshold, label, neg, pos, idx)

        # enable the grid and set it below other elements
        axis.set_axisbelow(True)
        axis.grid(True, color=self._grid_color)

        # if it was the last subplot of the page or the last subplot
        # to display, save figure
        if self._step_print == sub_plot_idx or (
            is_lower and sys == self.n_systems - 1
        ):
            self.finalize_one_page()
        return axis

    def _get_title(self, idx, dflt=None):
        """Get the histo title for the given idx"""
        title = (
            self._titles[idx]
            if self._titles is not None and idx < len(self._titles)
            else dflt
        )
        title = title or self._title_base
        title = (
            "" if title is not None and not title.replace(" ", "") else title
        )
        return title or ""

    def _get_neg_pos_thres(self, idx, input_scores, input_names):
        """Get scores and threshod for the given system at index idx"""
        neg_list, pos_list, _ = utils.get_fta_list(input_scores)
        length = len(neg_list)
        # lists returned by get_fta_list contains all the following items:
        # for bio or measure without eval:
        #   [dev]
        # for vuln with {licit,spoof} with eval:
        #   [dev, eval]
        # for vuln with {licit,spoof} without eval:
        #   [licit_dev, spoof_dev]
        # for vuln with {licit,spoof} with eval:
        #   [licit_dev, licit_eval, spoof_dev, spoof_eval]
        step = 2 if self._eval else 1
        # can have several files for one system
        dev_neg = [neg_list[x] for x in range(0, length, step)]
        dev_pos = [pos_list[x] for x in range(0, length, step)]
        eval_neg = eval_pos = None
        if self._eval:
            eval_neg = [neg_list[x] for x in range(1, length, step)]
            eval_pos = [pos_list[x] for x in range(1, length, step)]

        threshold = (
            utils.get_thres(self._criterion, dev_neg[0], dev_pos[0])
            if self._thres is None
            else self._thres[idx]
        )
        return dev_neg, dev_pos, eval_neg, eval_pos, threshold

    def _density_hist(self, scores, n, **kwargs):
        """Plots one density histo"""
        n, bins, patches = mpl.hist(
            scores, density=True, bins=self._nbins[n], **kwargs
        )
        return (n, bins, patches)

    def _lines(
        self, threshold, label=None, neg=None, pos=None, idx=None, **kwargs
    ):
        """Plots vertical line at threshold"""
        label = label or "Threshold"
        kwargs.setdefault("color", "C3")
        kwargs.setdefault("linestyle", "--")
        kwargs.setdefault("label", label)
        # plot a vertical threshold line
        mpl.axvline(x=threshold, ymin=0, ymax=1, **kwargs)

    def _setup_hist(self, neg, pos):
        """This function can be overwritten in derived classes

        Plots all the density histo required in one plot. Here negative and
        positive scores densities.
        """
        self._density_hist(
            neg[0], n=0, label="Negatives", alpha=0.5, color="C3"
        )
        self._density_hist(
            pos[0], n=1, label="Positives", alpha=0.5, color="C0"
        )
