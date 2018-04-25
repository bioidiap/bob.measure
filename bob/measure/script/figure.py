'''Runs error analysis on score sets, outputs metrics and plots'''

from __future__ import division, print_function
from abc import ABCMeta, abstractmethod
import sys
import os.path
import click
import matplotlib
import matplotlib.pyplot as mpl
from matplotlib.backends.backend_pdf import PdfPages
from tabulate import tabulate
from .. import (far_threshold, plot, utils, ppndf)

LINESTYLES = [
    (0, ()),                    #solid
    (0, (4, 4)),                #dashed
    (0, (1, 5)),                #dotted
    (0, (3, 5, 1, 5)),          #dashdotted
    (0, (3, 5, 1, 5, 1, 5)),    #dashdotdotted
    (0, (5, 1)),                #densely dashed
    (0, (1, 1)),                #densely dotted
    (0, (3, 1, 1, 1)),          #densely dashdotted
    (0, (3, 1, 1, 1, 1, 1)),    #densely dashdotdotted
    (0, (5, 10)),               #loosely dashed
    (0, (3, 10, 1, 10)),        #loosely dashdotted
    (0, (3, 10, 1, 10, 1, 10)), #loosely dashdotdotted
    (0, (1, 10))                #loosely dotted
]

class MeasureBase(object):
    """Base class for metrics and plots.
    This abstract class define the framework to plot or compute metrics from a
    list of (positive, negative) scores tuples.

    Attributes
    ----------
    func_load:
        Function that is used to load the input files
    """
    __metaclass__ = ABCMeta #for python 2.7 compatibility
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
        self._min_arg = 1 if 'min_arg' not in ctx.meta else ctx.meta['min_arg']
        self._ctx = ctx
        self.func_load = func_load
        self._titles = None if 'titles' not in ctx.meta else ctx.meta['titles']
        self._eval = evaluation
        self._min_arg = 1 if 'min_arg' not in ctx.meta else ctx.meta['min_arg']
        if len(scores) < 1 or len(scores) % self._min_arg != 0:
            raise click.BadParameter(
                'Number of argument must be a non-zero multiple of %d' % self._min_arg
            )
        self.n_systems = int(len(scores) / self._min_arg)
        if self._titles is not None and len(self._titles) != self.n_systems:
            raise click.BadParameter("Number of titles must be equal to the "
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
        #init matplotlib, log files, ...
        self.init_process()
        #iterates through the different systems and feed `compute`
        #with the dev (and eval) scores of each system
        # Note that more than one dev or eval scores score can be passed to
        # each system
        for idx in range(self.n_systems):
            input_scores, input_names = self._load_files(
                self._scores[idx:(idx + self._min_arg)]
            )
            self.compute(idx, input_scores, input_names)
        #setup final configuration, plotting properties, ...
        self.end_process()

    #protected functions that need to be overwritten
    def init_process(self):
        """ Called in :py:func:`~bob.measure.script.figure.MeasureBase`.run
        before iterating through the different systems.
        Should reimplemented in derived classes"""
        pass

    #Main computations are done here in the subclasses
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

    #Things to do after the main iterative computations are done
    @abstractmethod
    def end_process(self):
        """ Called in :py:func:`~bob.measure.script.figure.MeasureBase`.run
        after iterating through the different systems.
        Should reimplemented in derived classes"""
        pass

    #common protected functions

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
        self._tablefmt = None if 'tablefmt' not in ctx.meta else\
                ctx.meta['tablefmt']
        self._criter = None if 'criter' not in ctx.meta else ctx.meta['criter']
        self._open_mode = None if 'open_mode' not in ctx.meta else\
                ctx.meta['open_mode']
        self._thres = None if 'thres' not in ctx.meta else ctx.meta['thres']
        if self._thres is not None :
            if len(self._thres) == 1:
                self._thres = self._thres * self.n_systems
            elif len(self._thres) != self.n_systems:
                raise click.BadParameter(
                    '#thresholds must be the same as #systems (%d)' \
                    % len(self.n_systems)
                )
        self._far = None if 'far_value' not in ctx.meta else \
        ctx.meta['far_value']
        self._log = None if 'log' not in ctx.meta else ctx.meta['log']
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

        threshold = utils.get_thres(self._criter, dev_neg, dev_pos, self._far) \
                if self._thres is None else self._thres[idx]
        title = self._titles[idx] if self._titles is not None else None
        if self._thres is None:
            far_str = ''
            if self._criter == 'far' and self._far is not None:
                far_str = str(self._far)
            click.echo("[Min. criterion: %s %s] Threshold on Development set `%s`: %e"\
                       % (self._criter.upper(), far_str, title or dev_file, threshold),
                       file=self.log_file)
        else:
            click.echo("[Min. criterion: user provider] Threshold on "
                       "Development set `%s`: %e"\
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
            eval_fnm = int(round(eval_fnmr * eval_nc))  # number of false rejects

            eval_fta_str = "%.1f%%" % (100 * eval_fta)
            eval_fmr_str = "%.1f%% (%d/%d)" % (100 * eval_fmr, eval_fm, eval_ni)
            eval_fnmr_str = "%.1f%% (%d/%d)" % (100 * eval_fnmr, eval_fnm, eval_nc)

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
        self._output = None if 'output' not in ctx.meta else ctx.meta['output']
        self._points = 100 if 'points' not in ctx.meta else ctx.meta['points']
        self._split = None if 'split' not in ctx.meta else ctx.meta['split']
        self._axlim = None if 'axlim' not in ctx.meta else ctx.meta['axlim']
        self._clayout = None if 'clayout' not in ctx.meta else\
        ctx.meta['clayout']
        self._far_at = None if 'lines_at' not in ctx.meta else\
        ctx.meta['lines_at']
        self._trans_far_val = self._far_at
        if self._far_at is not None:
            self._eval_points = {line: [] for line in self._far_at}
            self._lines_val = []
        self._print_fn = True if 'show_fn' not in ctx.meta else\
        ctx.meta['show_fn']
        self._x_rotation = None if 'x_rotation' not in ctx.meta else \
                ctx.meta['x_rotation']
        if 'style' in ctx.meta:
            mpl.style.use(ctx.meta['style'])
        self._nb_figs = 2 if self._eval and self._split else 1
        self._colors = utils.get_colors(self.n_systems)
        self._states = ['Development', 'Evaluation']
        self._title = None if 'title' not in ctx.meta else ctx.meta['title']
        self._x_label = None if 'x_label' not in ctx.meta else\
        ctx.meta['x_label']
        self._y_label = None if 'y_label' not in ctx.meta else\
        ctx.meta['y_label']
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
            fs = None if 'figsize' not in self._ctx.meta else\
                    self._ctx.meta['figsize']
            fig = mpl.figure(i + 1, figsize=fs)
            fig.set_constrained_layout(self._clayout)
            fig.clear()

    def end_process(self):
        ''' Set title, legend, axis labels, grid colors, save figures, drow
        lines and close pdf if needed '''
        #draw vertical lines
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
        #only for plots
        if self._end_setup_plot:
            for i in range(self._nb_figs):
                fig = mpl.figure(i + 1)
                title = self._title
                if not self._eval:
                    title += (" (%s)" % self._states[0])
                elif self._split:
                    title += (" (%s)" % self._states[i])
                mpl.title(title)
                mpl.xlabel(self._x_label)
                mpl.ylabel(self._y_label)
                mpl.grid(True, color=self._grid_color)
                mpl.legend(loc='best')
                self._set_axis()
                mpl.xticks(rotation=self._x_rotation)
                self._pdf_page.savefig(fig)

        #do not want to close PDF when running evaluate
        if 'PdfPages' in self._ctx.meta and \
           ('closef' not in self._ctx.meta or self._ctx.meta['closef']):
            self._pdf_page.close()

    #common protected functions

    def _label(self, base, name, idx):
        if self._titles is not None and len(self._titles) > idx:
            return self._titles[idx]
        if self.n_systems > 1:
            return base + (" %d (%s)" % (idx + 1, name))
        return base + (" (%s)" % name)

    def _set_axis(self):
        if self._axlim is not None and None not in self._axlim:
            mpl.axis(self._axlim)

class Roc(PlotBase):
    ''' Handles the plotting of ROC'''
    def __init__(self, ctx, scores, evaluation, func_load):
        super(Roc, self).__init__(ctx, scores, evaluation, func_load)
        self._title = self._title or 'ROC'
        self._x_label = self._x_label or 'False Positive Rate'
        self._y_label = self._y_label or "1 - False Negative Rate"
        #custom defaults
        if self._axlim is None:
            self._axlim = [1e-4, 1.0, 1e-4, 1.0]

    def compute(self, idx, input_scores, input_names):
        ''' Plot ROC for dev and eval data using
        :py:func:`bob.measure.plot.roc`'''
        neg_list, pos_list, fta_list = utils.get_fta_list(input_scores)
        dev_neg, dev_pos, _ = neg_list[0], pos_list[0], fta_list[0]
        dev_file = input_names[0]
        if self._eval:
            eval_neg, eval_pos, _ = neg_list[1], pos_list[1], fta_list[1]
            eval_file = input_names[1]

        mpl.figure(1)
        if self._eval:
            linestyle = '-' if not self._split else LINESTYLES[idx % 14]
            plot.roc_for_far(
                dev_neg, dev_pos,
                color=self._colors[idx], linestyle=linestyle,
                label=self._label('development', dev_file, idx)
            )
            linestyle = '--'
            if self._split:
                mpl.figure(2)
                linestyle = LINESTYLES[idx % 14]

            plot.roc_for_far(
                eval_neg, eval_pos,
                color=self._colors[idx], linestyle=linestyle,
                label=self._label('eval', eval_file, idx)
            )
            if self._far_at is not None:
                from .. import farfrr
                for line in self._far_at:
                    thres_line = far_threshold(dev_neg, dev_pos, line)
                    eval_fmr, eval_fnmr = farfrr(eval_neg, eval_pos, thres_line)
                    eval_fnmr = 1 - eval_fnmr
                    mpl.scatter(eval_fmr, eval_fnmr, c=self._colors[idx], s=30)
                    self._eval_points[line].append((eval_fmr, eval_fnmr))
        else:
            plot.roc_for_far(
                dev_neg, dev_pos,
                color=self._colors[idx], linestyle=LINESTYLES[idx % 14],
                label=self._label('development', dev_file, idx)
            )

class Det(PlotBase):
    ''' Handles the plotting of DET '''
    def __init__(self, ctx, scores, evaluation, func_load):
        super(Det, self).__init__(ctx, scores, evaluation, func_load)
        self._title = self._title or 'DET'
        self._x_label = self._x_label or 'False Positive Rate'
        self._y_label = self._y_label or 'False Negative Rate'
        if self._far_at is not None:
            self._trans_far_val = [ppndf(float(k)) for k in self._far_at]
        #custom defaults here
        if self._x_rotation is None:
            self._x_rotation = 50

    def compute(self, idx, input_scores, input_names):
        ''' Plot DET for dev and eval data using
        :py:func:`bob.measure.plot.det`'''
        neg_list, pos_list, fta_list = utils.get_fta_list(input_scores)
        dev_neg, dev_pos, _ = neg_list[0], pos_list[0], fta_list[0]
        dev_file = input_names[0]
        if self._eval:
            eval_neg, eval_pos, _ = neg_list[1], pos_list[1], fta_list[1]
            eval_file = input_names[1]

        mpl.figure(1)
        if self._eval and eval_neg is not None:
            linestyle = '-' if not self._split else LINESTYLES[idx % 14]
            plot.det(
                dev_neg, dev_pos, self._points, color=self._colors[idx],
                linestyle=linestyle,
                label=self._label('development', dev_file, idx)
            )
            if self._split:
                mpl.figure(2)
            linestyle = '--' if not self._split else LINESTYLES[idx % 14]
            plot.det(
                eval_neg, eval_pos, self._points, color=self._colors[idx],
                linestyle=linestyle,
                label=self._label('eval', eval_file, idx)
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
            plot.det(
                dev_neg, dev_pos, self._points, color=self._colors[idx],
                linestyle=LINESTYLES[idx % 14],
                label=self._label('development', dev_file, idx)
            )

    def _set_axis(self):
        if self._axlim is not None and None not in self._axlim:
            plot.det_axis(self._axlim)
        else:
            plot.det_axis([0.01, 99, 0.01, 99])

class Epc(PlotBase):
    ''' Handles the plotting of EPC '''
    def __init__(self, ctx, scores, evaluation, func_load):
        super(Epc, self).__init__(ctx, scores, evaluation, func_load)
        if self._min_arg != 2:
            raise click.UsageError("EPC requires dev and eval score files")
        self._title = self._title or 'EPC'
        self._x_label = self._x_label or r'$\alpha$'
        self._y_label = self._y_label or 'HTER (%)'
        self._eval = True #always eval data with EPC
        self._split = False
        self._nb_figs = 1
        self._far_at = None

    def compute(self, idx, input_scores, input_names):
        ''' Plot EPC using :py:func:`bob.measure.plot.epc` '''
        neg_list, pos_list, fta_list = utils.get_fta_list(input_scores)
        dev_neg, dev_pos, _ = neg_list[0], pos_list[0], fta_list[0]
        dev_file = input_names[0]
        if self._eval:
            eval_neg, eval_pos, _ = neg_list[1], pos_list[1], fta_list[1]
            eval_file = input_names[1]

        plot.epc(
            dev_neg, dev_pos, eval_neg, eval_pos, self._points,
            color=self._colors[idx], linestyle=LINESTYLES[idx % 14],
            label=self._label(
                'curve', dev_file + "_" + eval_file, idx
            )
        )

class Hist(PlotBase):
    ''' Functional base class for histograms'''
    def __init__(self, ctx, scores, evaluation, func_load):
        super(Hist, self).__init__(ctx, scores, evaluation, func_load)
        self._nbins = [] if 'n_bins' not in ctx.meta else ctx.meta['n_bins']
        self._thres = None if 'thres' not in ctx.meta else ctx.meta['thres']
        self._show_dev = ((not self._eval) if 'show_dev' not in ctx.meta else\
                ctx.meta['show_dev']) or not self._eval
        if self._thres is not None and len(self._thres) != self.n_systems:
            if len(self._thres) == 1:
                self._thres = self._thres * self.n_systems
            else:
                raise click.BadParameter(
                    '#thresholds must be the same as #systems (%d)' \
                    % self.n_systems
                )
        self._criter = None if 'criter' not in ctx.meta else ctx.meta['criter']
        self._y_label = 'Dev. probability density' if self._eval else \
                'density' or self._y_label
        self._x_label = 'Scores' if not self._eval else ''
        self._title_base = self._title or 'Scores'
        self._end_setup_plot = False

    def compute(self, idx, input_scores, input_names):
        ''' Draw histograms of negative and positive scores.'''
        dev_neg, dev_pos, eval_neg, eval_pos, threshold = \
        self._get_neg_pos_thres(idx, input_scores, input_names)
        dev_file = input_names[0]
        eval_file = None if len(input_names) != 2 else input_names[1]

        fig = mpl.figure()
        if eval_neg is not None and self._show_dev:
            mpl.subplot(2, 1, 1)
        if self._show_dev:
            self._setup_hist(dev_neg, dev_pos)
            mpl.title(self._get_title(idx, dev_file, eval_file))
            mpl.ylabel(self._y_label)
            mpl.xlabel(self._x_label)
            if eval_neg is not None and self._show_dev:
                ax = mpl.gca()
                ax.axes.get_xaxis().set_ticklabels([])
            #Setup lines, corresponding axis and legends
            self._lines(threshold, dev_neg, dev_pos)
            if self._eval:
                self._plot_legends()

        if eval_neg is not None:
            if self._show_dev:
                mpl.subplot(2, 1, 2)
            self._setup_hist(
                eval_neg, eval_pos
            )
            if not self._show_dev:
                mpl.title(self._get_title(idx, dev_file, eval_file))
            mpl.ylabel('Eval. probability density')
            mpl.xlabel(self._x_label)
            #Setup lines, corresponding axis and legends
            self._lines(threshold, eval_neg, eval_pos)
            if not self._show_dev:
                self._plot_legends()

        self._pdf_page.savefig(fig)

    def _get_title(self, idx, dev_file, eval_file):
        title = self._titles[idx] if self._titles is not None else None
        if title is None:
            title = self._title_base if not self._print_fn else \
                    ('%s \n (%s)' % (
                        self._title_base,
                        str(dev_file) + (" / %s" % str(eval_file) if self._eval else "")
                    ))
        return title

    def _plot_legends(self):
        lines = []
        labels = []
        for ax in mpl.gcf().get_axes():
            li, la = ax.get_legend_handles_labels()
            lines += li
            labels += la
        if self._show_dev and self._eval:
            mpl.legend(
                lines, labels, loc='upper center', ncol=6,
                bbox_to_anchor=(0.5, -0.01), fontsize=6
            )
        else:
            mpl.legend(lines, labels,
                       loc='best', fancybox=True, framealpha=0.5)

    def _get_neg_pos_thres(self, idx, input_scores, input_names):
        neg_list, pos_list, _ = utils.get_fta_list(input_scores)
        length = len(neg_list)
        #can have several files for one system
        dev_neg = [neg_list[x] for x in range(0, length, 2)]
        dev_pos = [pos_list[x] for x in range(0, length, 2)]
        eval_neg = eval_pos = None
        if self._eval:
            eval_neg = [neg_list[x] for x in range(1, length, 2)]
            eval_pos = [pos_list[x] for x in range(1, length, 2)]

        threshold = utils.get_thres(
            self._criter, dev_neg[0], dev_pos[0]
        ) if self._thres is None else self._thres[idx]
        return dev_neg, dev_pos, eval_neg, eval_pos, threshold

    def _density_hist(self, scores, n, **kwargs):
        n, bins, patches = mpl.hist(
            scores, density=True,
            bins='auto' if len(self._nbins) <= n else self._nbins[n],
            **kwargs
        )
        return (n, bins, patches)

    def _lines(self, threshold, neg=None, pos=None, **kwargs):
        label = 'Threshold' if self._criter is None else self._criter.upper()
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

