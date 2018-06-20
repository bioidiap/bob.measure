'''Runs error analysis on score sets, outputs metrics and plots'''

from __future__ import division, print_function
from abc import ABCMeta, abstractmethod
import math
import sys
import os.path
import click
import matplotlib
import matplotlib.pyplot as mpl
from matplotlib.backends.backend_pdf import PdfPages
from tabulate import tabulate
from .. import (far_threshold, plot, utils, ppndf)


def check_list_value(values, desired_number, name, name2='systems'):
    if values is not None and len(values) != desired_number:
        if len(values) == 1:
            values = values * desired_number
        else:
            raise click.BadParameter(
                '#{} ({}) must be either 1 value or the same as '
                '#{} ({} values)'.format(name, values, name2, desired_number))

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
        self._min_arg = ctx.meta.get('min_arg', 1)
        self._ctx = ctx
        self.func_load = func_load
        self._legends = ctx.meta.get('legends')
        self._eval = evaluation
        self._min_arg = ctx.meta.get('min_arg', 1)
        if len(scores) < 1 or len(scores) % self._min_arg != 0:
            raise click.BadParameter(
                'Number of argument must be a non-zero multiple of %d' % self._min_arg
            )
        self.n_systems = int(len(scores) / self._min_arg)
        if self._legends is not None and len(self._legends) != self.n_systems:
            raise click.BadParameter("Number of legends must be equal to the "
                                     "number of systems")

    def run(self):
        """ Generate outputs (e.g. metrics, files, pdf plots).
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
                self._scores[idx * self._min_arg:(idx + 1) * self._min_arg]
            )
            self.compute(idx, input_scores, input_names)
        # setup final configuration, plotting properties, ...
        self.end_process()

    # protected functions that need to be overwritten
    def init_process(self):
        """ Called in :py:func:`~bob.measure.script.figure.MeasureBase`.run
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

    # Things to do after the main iterative computations are done
    @abstractmethod
    def end_process(self):
        """ Called in :py:func:`~bob.measure.script.figure.MeasureBase`.run
        after iterating through the different systems.
        Should reimplemented in derived classes"""
        pass

    # common protected functions

    def _load_files(self, filepaths):
        ''' Load the input files and return the base names of the files

        Returns
        -------
            scores: :any:`list`:
                A list that contains the output of
                ``func_load`` for the given files
            basenames: :any:`list`:
                A list of basenames for the given files
        '''
        scores = []
        basenames = []
        for filename in filepaths:
            basenames.append(os.path.basename(filename).split(".")[0])
            scores.append(self.func_load(filename))
        return scores, basenames


class Metrics(MeasureBase):
    ''' Compute metrics from score files

    Attributes
    ----------
    log_file: str
        output stream
    '''

    def __init__(self, ctx, scores, evaluation, func_load):
        super(Metrics, self).__init__(ctx, scores, evaluation, func_load)
        self._tablefmt = ctx.meta.get('tablefmt')
        self._criterion = ctx.meta.get('criterion')
        self._open_mode = ctx.meta.get('open_mode')
        self._thres = ctx.meta.get('thres')
        if self._thres is not None:
            if len(self._thres) == 1:
                self._thres = self._thres * self.n_systems
            elif len(self._thres) != self.n_systems:
                raise click.BadParameter(
                    '#thresholds must be the same as #systems (%d)'
                    % len(self.n_systems)
                )
        self._far = ctx.meta.get('far_value')
        self._log = ctx.meta.get('log')
        self.log_file = sys.stdout
        if self._log is not None:
            self.log_file = open(self._log, self._open_mode)

    def compute(self, idx, input_scores, input_names):
        ''' Compute metrics thresholds and tables (FAR, FMR, FNMR, HTER) for
        given system inputs'''
        neg_list, pos_list, fta_list = utils.get_fta_list(input_scores)
        dev_neg, dev_pos, dev_fta = neg_list[0], pos_list[0], fta_list[0]
        dev_file = input_names[0]
        if self._eval:
            eval_neg, eval_pos, eval_fta = neg_list[1], pos_list[1], fta_list[1]
            eval_file = input_names[1]

        threshold = utils.get_thres(self._criterion, dev_neg, dev_pos, self._far) \
            if self._thres is None else self._thres[idx]
        title = self._legends[idx] if self._legends is not None else None
        if self._thres is None:
            far_str = ''
            if self._criterion == 'far' and self._far is not None:
                far_str = str(self._far)
            click.echo("[Min. criterion: %s %s] Threshold on Development set `%s`: %e"
                       % (self._criterion.upper(),
                          far_str, title or dev_file,
                          threshold),
                       file=self.log_file)
        else:
            click.echo("[Min. criterion: user provided] Threshold on "
                       "Development set `%s`: %e"
                       % (dev_file or title, threshold), file=self.log_file)

        from .. import farfrr
        dev_fmr, dev_fnmr = farfrr(dev_neg, dev_pos, threshold)
        dev_far = dev_fmr * (1 - dev_fta)
        dev_frr = dev_fta + dev_fnmr * (1 - dev_fta)
        dev_hter = (dev_far + dev_frr) / 2.0

        dev_ni = dev_neg.shape[0]  # number of impostors
        dev_fm = int(round(dev_fmr * dev_ni))  # number of false accepts
        dev_nc = dev_pos.shape[0]  # number of clients
        dev_fnm = int(round(dev_fnmr * dev_nc))  # number of false rejects

        dev_fta_str = "%.1f%%" % (100 * dev_fta)
        dev_fmr_str = "%.1f%% (%d/%d)" % (100 * dev_fmr, dev_fm, dev_ni)
        dev_fnmr_str = "%.1f%% (%d/%d)" % (100 * dev_fnmr, dev_fnm, dev_nc)
        dev_far_str = "%.1f%%" % (100 * dev_far)
        dev_frr_str = "%.1f%%" % (100 * dev_frr)
        dev_hter_str = "%.1f%%" % (100 * dev_hter)
        headers = ['' or title, 'Development %s' % dev_file]
        raws = [['FtA', dev_fta_str],
                ['FMR', dev_fmr_str],
                ['FNMR', dev_fnmr_str],
                ['FAR', dev_far_str],
                ['FRR', dev_frr_str],
                ['HTER', dev_hter_str]]

        if self._eval:
            # computes statistics for the eval set based on the threshold a priori
            eval_fmr, eval_fnmr = farfrr(eval_neg, eval_pos, threshold)
            eval_far = eval_fmr * (1 - eval_fta)
            eval_frr = eval_fta + eval_fnmr * (1 - eval_fta)
            eval_hter = (eval_far + eval_frr) / 2.0

            eval_ni = eval_neg.shape[0]  # number of impostors
            eval_fm = int(round(eval_fmr * eval_ni))  # number of false accepts
            eval_nc = eval_pos.shape[0]  # number of clients
            # number of false rejects
            eval_fnm = int(round(eval_fnmr * eval_nc))

            eval_fta_str = "%.1f%%" % (100 * eval_fta)
            eval_fmr_str = "%.1f%% (%d/%d)" % (100 *
                                               eval_fmr, eval_fm, eval_ni)
            eval_fnmr_str = "%.1f%% (%d/%d)" % (100 *
                                                eval_fnmr, eval_fnm, eval_nc)

            eval_far_str = "%.1f%%" % (100 * eval_far)
            eval_frr_str = "%.1f%%" % (100 * eval_frr)
            eval_hter_str = "%.1f%%" % (100 * eval_hter)

            headers.append('Eval. % s' % eval_file)
            raws[0].append(eval_fta_str)
            raws[1].append(eval_fmr_str)
            raws[2].append(eval_fnmr_str)
            raws[3].append(eval_far_str)
            raws[4].append(eval_frr_str)
            raws[5].append(eval_hter_str)

        click.echo(tabulate(raws, headers, self._tablefmt), file=self.log_file)

    def end_process(self):
        ''' Close log file if needed'''
        if self._log is not None:
            self.log_file.close()


class PlotBase(MeasureBase):
    ''' Base class for plots. Regroup several options and code
    shared by the different plots
    '''

    def __init__(self, ctx, scores, evaluation, func_load):
        super(PlotBase, self).__init__(ctx, scores, evaluation, func_load)
        self._output = ctx.meta.get('output')
        self._points = ctx.meta.get('points', 100)
        self._split = ctx.meta.get('split')
        self._axlim = ctx.meta.get('axlim')
        self._disp_legend = ctx.meta.get('disp_legend', True)
        self._legend_loc = ctx.meta.get('legend_loc')
        self._min_dig = None
        if 'min_far_value' in ctx.meta:
            self._min_dig = int(math.log10(ctx.meta['min_far_value']))
        elif self._axlim is not None and self._axlim[0] is not None:
            self._min_dig = int(math.log10(self._axlim[0])
                                if self._axlim[0] != 0 else 0)
        self._clayout = ctx.meta.get('clayout')
        self._far_at = ctx.meta.get('lines_at')
        self._trans_far_val = self._far_at
        if self._far_at is not None:
            self._eval_points = {line: [] for line in self._far_at}
            self._lines_val = []
        self._print_fn = ctx.meta.get('show_fn', True)
        self._x_rotation = ctx.meta.get('x_rotation')
        if 'style' in ctx.meta:
            mpl.style.use(ctx.meta['style'])
        self._nb_figs = 2 if self._eval and self._split else 1
        self._colors = utils.get_colors(self.n_systems)
        self._line_linestyles = ctx.meta.get('line_linestyles', False)
        self._linestyles = utils.get_linestyles(
            self.n_systems, self._line_linestyles)
        self._titles = ctx.meta.get('titles', []) * 2
        # for compatibility
        self._title = ctx.meta.get('title')
        if not self._titles and self._title is not None:
            self._titles = [self._title] * 2

        self._x_label = ctx.meta.get('x_label')
        self._y_label = ctx.meta.get('y_label')
        self._grid_color = 'silver'
        self._pdf_page = None
        self._end_setup_plot = True

    def init_process(self):
        ''' Open pdf and set axis font size if provided '''
        if not hasattr(matplotlib, 'backends'):
            matplotlib.use('pdf')

        self._pdf_page = self._ctx.meta['PdfPages'] if 'PdfPages'in \
            self._ctx.meta else PdfPages(self._output)

        for i in range(self._nb_figs):
            fs = self._ctx.meta.get('figsize')
            fig = mpl.figure(i + 1, figsize=fs)
            fig.set_constrained_layout(self._clayout)
            fig.clear()

    def end_process(self):
        ''' Set title, legend, axis labels, grid colors, save figures, drow
        lines and close pdf if needed '''
        # draw vertical lines
        if self._far_at is not None:
            for (line, line_trans) in zip(self._far_at, self._trans_far_val):
                mpl.figure(1)
                mpl.plot(
                    [line_trans, line_trans], [-100.0, 100.], "--",
                    color='black'
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
                    mpl.plot(x_values,
                             y_values, '--',
                             color='black')
        # only for plots
        if self._end_setup_plot:
            for i in range(self._nb_figs):
                fig = mpl.figure(i + 1)
                title = '' if not self._titles else self._titles[i]
                mpl.title(title if title.replace(' ', '') else '')
                mpl.xlabel(self._x_label)
                mpl.ylabel(self._y_label)
                mpl.grid(True, color=self._grid_color)
                if self._disp_legend:
                    mpl.legend(loc=self._legend_loc)
                self._set_axis()
                mpl.xticks(rotation=self._x_rotation)
                self._pdf_page.savefig(fig)

        # do not want to close PDF when running evaluate
        if 'PdfPages' in self._ctx.meta and \
           ('closef' not in self._ctx.meta or self._ctx.meta['closef']):
            self._pdf_page.close()

    # common protected functions

    def _label(self, base, name, idx):
        if self._legends is not None and len(self._legends) > idx:
            return self._legends[idx]
        if self.n_systems > 1:
            return base + (" %d (%s)" % (idx + 1, name))
        return base + (" (%s)" % name)

    def _set_axis(self):
        if self._axlim is not None:
            mpl.axis(self._axlim)


class Roc(PlotBase):
    ''' Handles the plotting of ROC'''

    def __init__(self, ctx, scores, evaluation, func_load):
        super(Roc, self).__init__(ctx, scores, evaluation, func_load)
        self._titles = self._titles or ['ROC dev', 'ROC eval']
        self._x_label = self._x_label or 'False Positive Rate'
        self._y_label = self._y_label or "1 - False Negative Rate"
        self._semilogx = ctx.meta.get('semilogx', True)
        best_legend = 'lower right' if self._semilogx else 'upper right'
        self._legend_loc = self._legend_loc or best_legend
        # custom defaults
        if self._axlim is None:
            self._axlim = [None, None, -0.05, 1.05]

    def compute(self, idx, input_scores, input_names):
        ''' Plot ROC for dev and eval data using
        :py:func:`bob.measure.plot.roc`'''
        neg_list, pos_list, _ = utils.get_fta_list(input_scores)
        dev_neg, dev_pos = neg_list[0], pos_list[0]
        dev_file = input_names[0]
        if self._eval:
            eval_neg, eval_pos = neg_list[1], pos_list[1]
            eval_file = input_names[1]

        mpl.figure(1)
        if self._eval:
            plot.roc_for_far(
                dev_neg, dev_pos,
                far_values=plot.log_values(self._min_dig or -4),
                CAR=self._semilogx,
                color=self._colors[idx], linestyle=self._linestyles[idx],
                label=self._label('dev', dev_file, idx)
            )
            if self._split:
                mpl.figure(2)

            linestyle = '--' if not self._split else self._linestyles[idx]
            plot.roc_for_far(
                eval_neg, eval_pos, linestyle=linestyle,
                far_values=plot.log_values(self._min_dig or -4),
                CAR=self._semilogx,
                color=self._colors[idx],
                label=self._label('eval', eval_file, idx)
            )
            if self._far_at is not None:
                from .. import farfrr
                for line in self._far_at:
                    thres_line = far_threshold(dev_neg, dev_pos, line)
                    eval_fmr, eval_fnmr = farfrr(
                        eval_neg, eval_pos, thres_line)
                    eval_fnmr = 1 - eval_fnmr
                    mpl.scatter(eval_fmr, eval_fnmr, c=self._colors[idx], s=30)
                    self._eval_points[line].append((eval_fmr, eval_fnmr))
        else:
            plot.roc_for_far(
                dev_neg, dev_pos,
                far_values=plot.log_values(self._min_dig or -4),
                CAR=self._semilogx,
                color=self._colors[idx], linestyle=self._linestyles[idx],
                label=self._label('dev', dev_file, idx)
            )


class Det(PlotBase):
    ''' Handles the plotting of DET '''

    def __init__(self, ctx, scores, evaluation, func_load):
        super(Det, self).__init__(ctx, scores, evaluation, func_load)
        self._titles = self._titles or ['DET dev', 'DET eval']
        self._x_label = self._x_label or 'False Positive Rate (%)'
        self._y_label = self._y_label or 'False Negative Rate (%)'
        self._legend_loc = self._legend_loc or 'upper right'
        if self._far_at is not None:
            self._trans_far_val = [ppndf(float(k)) for k in self._far_at]
        # custom defaults here
        if self._x_rotation is None:
            self._x_rotation = 50

        if self._axlim is None:
            self._axlim = [0.01, 99, 0.01, 99]

        if self._min_dig is not None:
            self._axlim[0] = math.pow(10, self._min_dig) * 100

    def compute(self, idx, input_scores, input_names):
        ''' Plot DET for dev and eval data using
        :py:func:`bob.measure.plot.det`'''
        neg_list, pos_list, _ = utils.get_fta_list(input_scores)
        dev_neg, dev_pos = neg_list[0], pos_list[0]
        dev_file = input_names[0]
        if self._eval:
            eval_neg, eval_pos = neg_list[1], pos_list[1]
            eval_file = input_names[1]

        mpl.figure(1)
        if self._eval and eval_neg is not None:
            plot.det(
                dev_neg, dev_pos, self._points, color=self._colors[idx],
                linestyle=self._linestyles[idx],
                label=self._label('development', dev_file, idx)
            )
            if self._split:
                mpl.figure(2)
            linestyle = '--' if not self._split else self._linestyles[idx]
            plot.det(
                eval_neg, eval_pos, self._points, color=self._colors[idx],
                linestyle=linestyle,
                label=self._label('eval', eval_file, idx)
            )
            if self._far_at is not None:
                from .. import farfrr
                for line in self._far_at:
                    thres_line = far_threshold(dev_neg, dev_pos, line)
                    eval_fmr, eval_fnmr = farfrr(
                        eval_neg, eval_pos, thres_line)
                    eval_fmr, eval_fnmr = ppndf(eval_fmr), ppndf(eval_fnmr)
                    mpl.scatter(eval_fmr, eval_fnmr, c=self._colors[idx], s=30)
                    self._eval_points[line].append((eval_fmr, eval_fnmr))
        else:
            plot.det(
                dev_neg, dev_pos, self._points, color=self._colors[idx],
                linestyle=self._linestyles[idx],
                label=self._label('development', dev_file, idx)
            )

    def _set_axis(self):
        plot.det_axis(self._axlim)


class Epc(PlotBase):
    ''' Handles the plotting of EPC '''

    def __init__(self, ctx, scores, evaluation, func_load):
        super(Epc, self).__init__(ctx, scores, evaluation, func_load)
        if self._min_arg != 2:
            raise click.UsageError("EPC requires dev and eval score files")
        self._titles = self._titles or ['EPC'] * 2
        self._x_label = self._x_label or r'$\alpha$'
        self._y_label = self._y_label or 'HTER (%)'
        self._legend_loc = self._legend_loc or 'upper center'
        self._eval = True  # always eval data with EPC
        self._split = False
        self._nb_figs = 1
        self._far_at = None

    def compute(self, idx, input_scores, input_names):
        ''' Plot EPC using :py:func:`bob.measure.plot.epc` '''
        neg_list, pos_list, _ = utils.get_fta_list(input_scores)
        dev_neg, dev_pos = neg_list[0], pos_list[0]
        dev_file = input_names[0]
        if self._eval:
            eval_neg, eval_pos = neg_list[1], pos_list[1]
            eval_file = input_names[1]

        plot.epc(
            dev_neg, dev_pos, eval_neg, eval_pos, self._points,
            color=self._colors[idx], linestyle=self._linestyles[idx],
            label=self._label(
                'curve', dev_file + "_" + eval_file, idx
            )
        )


class Hist(PlotBase):
    ''' Functional base class for histograms'''

    def __init__(self, ctx, scores, evaluation, func_load, nhist_per_system=2):
        super(Hist, self).__init__(ctx, scores, evaluation, func_load)
        self._nbins = ctx.meta.get('n_bins', ['doane'])
        self._nhist_per_system = nhist_per_system
        self._nbins = check_list_value(
            self._nbins, self.n_systems * nhist_per_system, 'n_bins',
            'histograms')
        self._thres = ctx.meta.get('thres')
        self._thres = check_list_value(
            self._thres, self.n_systems, 'thresholds')
        self._criterion = ctx.meta.get('criterion')
        self._no_line = ctx.meta.get('no_line', False)
        self._nrows = ctx.meta.get('n_row', 1)
        self._ncols = ctx.meta.get('n_col', 1)
        self._nlegends = ctx.meta.get('legends_ncol', 10)
        self._legend_loc = self._legend_loc or 'upper center'
        self._step_print = int(self._nrows * self._ncols)
        self._title_base = 'Scores'
        self._y_label = 'Probability density'
        self._x_label = 'Scores values'
        self._end_setup_plot = False

    def compute(self, idx, input_scores, input_names):
        ''' Draw histograms of negative and positive scores.'''
        dev_neg, dev_pos, eval_neg, eval_pos, threshold = \
            self._get_neg_pos_thres(idx, input_scores, input_names)
        dev_file = input_names[0]
        eval_file = None if len(input_names) != 2 else input_names[1]
        n = idx % self._step_print
        col = n % self._ncols
        sub_plot_idx = n + 1
        axis = mpl.subplot(self._nrows, self._ncols, sub_plot_idx)
        neg = eval_neg if eval_neg is not None else dev_neg
        pos = eval_pos if eval_pos is not None else dev_pos
        self._setup_hist(neg, pos)
        if col == 0:
            axis.set_ylabel(self._y_label)
        # rest to be printed
        rest_print = self.n_systems - \
            int(idx / self._step_print) * self._step_print
        if n + self._ncols >= min(self._step_print, rest_print):
            axis.set_xlabel(self._x_label)
        axis.set_title(self._get_title(idx, dev_file, eval_file))
        label = "%s threshold%s" % (
            '' if self._criterion is None else
            self._criterion.upper(), ' (dev)' if self._eval else ''
        )
        if self._eval and not self._no_line:
            self._lines(threshold, label, neg, pos, idx)
        if sub_plot_idx == 1:
            self._plot_legends()
        if self._step_print == sub_plot_idx or idx == self.n_systems - 1:
            mpl.tight_layout()
            self._pdf_page.savefig(mpl.gcf(), bbox_inches='tight')
            mpl.clf()
            mpl.figure()

    def _get_title(self, idx, dev_file, eval_file):
        title = self._legends[idx] if self._legends is not None else None
        title = title or self._title_base
        title = '' if title is not None and not title.replace(
            ' ', '') else title
        return title or ''

    def _plot_legends(self):
        lines = []
        labels = []
        for ax in mpl.gcf().get_axes():
            li, la = ax.get_legend_handles_labels()
            lines += li
            labels += la
        if self._disp_legend:
            mpl.gcf().legend(
                lines, labels, loc=self._legend_loc, fancybox=True,
                framealpha=0.5, ncol=self._nlegends,
                bbox_to_anchor=(0.55, 1.06),
            )

    def _get_neg_pos_thres(self, idx, input_scores, input_names):
        neg_list, pos_list, _ = utils.get_fta_list(input_scores)
        length = len(neg_list)
        # can have several files for one system
        dev_neg = [neg_list[x] for x in range(0, length, 2)]
        dev_pos = [pos_list[x] for x in range(0, length, 2)]
        eval_neg = eval_pos = None
        if self._eval:
            eval_neg = [neg_list[x] for x in range(1, length, 2)]
            eval_pos = [pos_list[x] for x in range(1, length, 2)]

        threshold = utils.get_thres(
            self._criterion, dev_neg[0], dev_pos[0]
        ) if self._thres is None else self._thres[idx]
        return dev_neg, dev_pos, eval_neg, eval_pos, threshold

    def _density_hist(self, scores, n, **kwargs):
        n, bins, patches = mpl.hist(
            scores, density=True,
            bins=self._nbins[n],
            **kwargs
        )
        return (n, bins, patches)

    def _lines(self, threshold, label=None, neg=None, pos=None,
               idx=None, **kwargs):
        label = label or 'Threshold'
        kwargs.setdefault('color', 'C3')
        kwargs.setdefault('linestyle', '--')
        kwargs.setdefault('label', label)
        # plot a vertical threshold line
        mpl.axvline(x=threshold, ymin=0, ymax=1, **kwargs)

    def _setup_hist(self, neg, pos):
        ''' This function can be overwritten in derived classes'''
        self._density_hist(
            neg[0], n=0,
            label='Negatives', alpha=0.5, color='C3'
        )
        self._density_hist(
            pos[0], n=1,
            label='Positives', alpha=0.5, color='C0'
        )
