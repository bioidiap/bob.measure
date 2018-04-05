'''Runs error analysis on score sets, outputs metrics and plots'''

from __future__ import division, print_function
from abc import ABCMeta, abstractmethod
import sys
import ntpath
import numpy
import click
import matplotlib
import matplotlib.pyplot as mpl
from matplotlib.backends.backend_pdf import PdfPages
from tabulate import tabulate
from .. import plot
from .. import utils

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

    _scores: :any:`list`:
        List of input files (e.g. dev-{1, 2, 3}, {dev,test}-scores1

    _ctx : :py:class:`dict`
        Click context dictionary.

    _test : :py:class:`bool`
        True if test data are used

    func_load:
        Function that is used to load the input files

    """
    __metaclass__ = ABCMeta #for python 2.7 compatibility
    def __init__(self, ctx, scores, test, func_load):
        """
        Parameters
        ----------

        ctx : :py:class:`dict`
            Click context dictionary.

        scores : :any:`list`:
            List of input files (e.g. dev-{1, 2, 3}, {dev,test}-scores1
            {dev,test}-scores2)
        test : :py:class:`bool`
            True if test data are used
        func_load : Function that is used to load the input files
        """
        self._scores = scores
        self._ctx = ctx
        self.func_load = func_load
        self.dev_names, self.test_names, self.dev_scores, self.test_scores = \
                self._load_files()
        self._test = test

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
        #with the dev (and test) scores of each system
        for idx, (dev_score, dev_file) in enumerate(
                zip(self.dev_scores, self.dev_names)
        ):
            test_score = self.test_scores[idx] if self.test_scores is not None\
            else None
            test_file = None if self.test_names is None else self.test_names[idx]
            dev_neg, dev_pos, dev_fta, test_neg, test_pos, test_fta =\
                self._process_scores(dev_score, test_score)
            #does the main computations/plottings here
            self.compute(idx, dev_neg, dev_pos, dev_fta, dev_file,
                          test_neg, test_pos, test_fta, test_file)
        #setup final configuration, plotting properties, ...
        self.end_process()

    #protected functions that need to be overwritten
    def init_process(self):
        """ Called in :py:func:`~bob.measure.script.figure.MeasureBase`.run
        before iterating through the different sytems.
        Should reimplemented in derived classes"""
        pass

    #Main computations are done here in the subclasses
    @abstractmethod
    def compute(self, idx, dev_neg, dev_pos, dev_fta=None, dev_file=None,
                test_neg=None, test_pos=None, test_fta=None, test_file=None):
        """Compute metrics or plots from the inputs given by
        :py:func:`~bob.measure.script.figure.MeasureBase.run`.
        Should reimplemented in derived classes

        Parameters
        ----------

        idx : :obj:`int`
            index of the system

        dev_neg : :py:class:`numpy.ndarray`
            negative dev scores
        dev_pos : :py:class:`numpy.ndarray`
            positive dev scores
        dev_fta : :obj:`float`
            failure to acquire rate
        dev_file : str
            name of the dev file without extension
        test_neg : :py:class:`numpy.ndarray`
            negative test scores
        test_pos : :py:class:`numpy.ndarray`
            positive test scores
        test_fta : :obj:`float`
            failure to acquire rate for test scores
        test_file : str
            name of the test file without extension
        """
        pass

    #Things to do after the main iterative computations are done
    @abstractmethod
    def end_process(self):
        pass

    #common protected functions

    def _load_files(self):
        ''' Load the input files and returns

        Returns
        -------

            :any:`list`: A list of tuples, where each tuple contains the
            ``negative`` and ``positive`` scores for one probe of the database. Both
            ``negatives`` and ``positives`` can be either an 1D'''

        dev_paths = self._scores if 'dev-scores' not in self._ctx.meta else \
                self._ctx.meta['dev-scores']
        test_paths = None if 'test-scores' not in self._ctx.meta else \
                self._ctx.meta['test-scores']
        def _extract_file_names(filenames):
            if filenames is None:
                return None
            res = []
            for file_path in filenames:
                _, name = ntpath.split(file_path)
                res.append(name.split(".")[0])
            return res
        return (_extract_file_names(dev_paths), _extract_file_names(test_paths),
                self.func_load(dev_paths), self.func_load(test_paths))

    def _process_scores(self, dev_score, test_score):
        '''Process score files and return neg/pos/fta for test and dev'''
        dev_neg = dev_pos = dev_fta = test_neg = test_pos = test_fta = None
        if dev_score[0] is not None:
            dev_score, dev_fta = utils.get_fta(dev_score)
            dev_neg, dev_pos = dev_score
            if dev_neg is None:
                raise click.UsageError("While loading dev-score file")

        if self._test and test_score is not None and test_score[0] is not None:
            test_score, test_fta = utils.get_fta(test_score)
            test_neg, test_pos = test_score
            if test_neg is None:
                raise click.UsageError("While loading test-score file")

        return (dev_neg, dev_pos, dev_fta, test_neg, test_pos, test_fta)


class Metrics(MeasureBase):
    ''' Compute metrics from score files

    Attributes
    ----------

    _tablefmt: str
        Table format

    _criter: str
        Criterion to compute threshold, see :py:func:`bob.measure.utils.get_thres`

    _open_mode: str
        Open mode of the output file (e.g. `w`, `a+`)

    _thres: :obj:`float`
        If given, uses this threshold instead of computing it

    _log: str
        Path to output log file

    log_file: str
        output stream

    '''
    def __init__(self, ctx, scores, test, func_load):
        super(Metrics, self).__init__(ctx, scores, test, func_load)
        self._tablefmt = None if 'tablefmt' not in ctx.meta else\
                ctx.meta['tablefmt']
        self._criter = None if 'criter' not in ctx.meta else ctx.meta['criter']
        self._open_mode = None if 'open_mode' not in ctx.meta else\
                ctx.meta['open_mode']
        self._thres = None if 'thres' not in ctx.meta else ctx.meta['thres']
        self._log = None if 'log' not in ctx.meta else ctx.meta['log']
        self.log_file = sys.stdout
        if self._log is not None:
            self.log_file = open(self._log, self._open_mode)

    def compute(self, idx, dev_neg, dev_pos, dev_fta=None, dev_file=None,
                test_neg=None, test_pos=None, test_fta=None, test_file=None):
        ''' Compute metrics thresholds and tables (FAR, FMR, FMNR, HTER) for
        given system inputs'''
        threshold = utils.get_thres(self._criter, dev_neg, dev_pos) \
                if self._thres is None else self._thres
        if self._thres is None:
            click.echo("[Min. criterion: %s] Threshold on Development set `%s`: %e"\
                       % (self._criter.upper(), dev_file, threshold),
                       file=self.log_file)
        else:
            click.echo("[Min. criterion: user provider] Threshold on "
                       "Development set `%s`: %e"\
                       % (dev_file, threshold), file=self.log_file)


        from .. import farfrr
        dev_fmr, dev_fnmr = farfrr(dev_neg, dev_pos, threshold)
        dev_far = dev_fmr * (1 - dev_fta)
        dev_frr = dev_fta + dev_fnmr * (1 - dev_fta)
        dev_hter = (dev_far + dev_frr) / 2.0

        dev_ni = dev_neg.shape[0]  # number of impostors
        dev_fm = int(round(dev_fmr * dev_ni))  # number of false accepts
        dev_nc = dev_pos.shape[0]  # number of clients
        dev_fnm = int(round(dev_fnmr * dev_nc))  # number of false rejects

        dev_fmr_str = "%.3f%% (%d/%d)" % (100 * dev_fmr, dev_fm, dev_ni)
        dev_fnmr_str = "%.3f%% (%d/%d)" % (100 * dev_fnmr, dev_fnm, dev_nc)
        dev_far_str = "%.3f%%" % (100 * dev_far)
        dev_frr_str = "%.3f%%" % (100 * dev_frr)
        dev_hter_str = "%.3f%%" % (100 * dev_hter)
        headers = ['', 'Development %s' % dev_file]
        raws = [['FMR', dev_fmr_str],
                ['FNMR', dev_fnmr_str],
                ['FAR', dev_far_str],
                ['FRR', dev_frr_str],
                ['HTER', dev_hter_str]]

        if self._test and test_neg is not None:
            # computes statistics for the test set based on the threshold a priori
            test_fmr, test_fnmr = farfrr(test_neg, test_pos, threshold)
            test_far = test_fmr * (1 - test_fta)
            test_frr = test_fta + test_fnmr * (1 - test_fta)
            test_hter = (test_far + test_frr) / 2.0

            test_ni = test_neg.shape[0]  # number of impostors
            test_fm = int(round(test_fmr * test_ni))  # number of false accepts
            test_nc = test_pos.shape[0]  # number of clients
            test_fnm = int(round(test_fnmr * test_nc))  # number of false rejects

            test_fmr_str = "%.3f%% (%d/%d)" % (100 * test_fmr, test_fm, test_ni)
            test_fnmr_str = "%.3f%% (%d/%d)" % (100 * test_fnmr, test_fnm, test_nc)

            test_far_str = "%.3f%%" % (100 * test_far)
            test_frr_str = "%.3f%%" % (100 * test_frr)
            test_hter_str = "%.3f%%" % (100 * test_hter)

            headers.append('Test % s' % self.test_names[idx])
            raws[0].append(test_fmr_str)
            raws[1].append(test_fnmr_str)
            raws[2].append(test_far_str)
            raws[3].append(test_frr_str)
            raws[4].append(test_hter_str)

        click.echo(tabulate(raws, headers, self._tablefmt), file=self.log_file)

    def end_process(self):
        ''' Close log file if needed'''
        if self._log is not None:
            self.log_file.close()

class PlotBase(MeasureBase):
    ''' Base class for plots. Regroup several options and code
    shared by the different plots

    Attributes
    ----------

    _output: str
        Path to the output pdf file

    _points: :obj:`int`
        Number of points used to draw curves

    _titles: :any:`list`
        List of titles for each system (dev + (test) scores)

    _split: :obj:`bool`
        If False, dev and test curves will be printed on the some figure

    _min_x: :obj:`float`
        Minimum value for the X-axis

    _max_x: :obj:`float`
        Maximum value for the X-axis

    _min_y: :obj:`float`
        Minimum value for the Y-axis

    _max_y: :obj:`float`
        Maximum value for the Y-axis

    _x_rotation: :obj:`int`
        Rotation of the X axis labels

    _axisfontsize: :obj:`int`
        Axis font size
    '''
    def __init__(self, ctx, scores, test, func_load):
        super(PlotBase, self).__init__(ctx, scores, test, func_load)
        self._output = None if 'output' not in ctx.meta else ctx.meta['output']
        self._points = None if 'points' not in ctx.meta else ctx.meta['points']
        self._titles = None if 'titles' not in ctx.meta else ctx.meta['titles']
        self._split = None if 'split' not in ctx.meta else ctx.meta['split']
        self._min_x = None if 'min_x' not in ctx.meta else ctx.meta['min_x']
        self._min_y = None if 'min_y' not in ctx.meta else ctx.meta['min_y']
        self._max_x = None if 'max_x' not in ctx.meta else ctx.meta['max_x']
        self._max_y = None if 'max_y' not in ctx.meta else ctx.meta['max_y']
        self._x_rotation = None if 'x_rotation' not in ctx.meta else \
                ctx.meta['x_rotation']
        self._axisfontsize = None if 'fontsize' not in ctx.meta else \
                ctx.meta['fontsize']

        self._nb_figs = 2 if self._test and self._split else 1
        self._multi_plots = len(self.dev_scores) > 1
        self._colors = utils.get_colors(len(self.dev_scores))
        self._states = ['Development', 'Evaluation']
        self._title = ''
        self._x_label = 'FMR (%)'
        self._y_label = 'FNMR (%)'
        self._grid_color = 'silver'
        self._pdf_page = None
        self._end_setup_plot = True

    def init_process(self):
        ''' Open pdf and set axis font size if provided '''
        if not hasattr(matplotlib, 'backends'):
            matplotlib.use('pdf')

        self._pdf_page = self._ctx.meta['PdfPages'] if 'PdfPages'in \
        self._ctx.meta else PdfPages(self._output)

        mpl.figure(1)

        if self._axisfontsize is not None:
            mpl.rc('xtick', labelsize=self._axisfontsize)
            mpl.rc('ytick', labelsize=self._axisfontsize)

    def end_process(self):
        ''' Set title, legend, axis labels, grid colors, save figures and
        close pdf is needed '''
        #only for plots
        if self._end_setup_plot:
            for i in range(self._nb_figs):
                fig = mpl.figure(i + 1)
                title = self._title
                if not self._test:
                    title += (" (%s)" % self._states[0])
                elif self._split:
                    title += (" (%s)" % self._states[i])
                mpl.title(title)
                mpl.xlabel(self._x_label)
                mpl.ylabel(self._y_label)
                mpl.grid(True, color=self._grid_color)
                mpl.legend()
                self._set_axis()
                #gives warning when applied with mpl
                fig.set_tight_layout(True)
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
        if self._multi_plots:
            return base + (" %d (%s)" % (idx + 1, name))
        return base + (" (%s)" % name)

    def _set_axis(self):
        axis = [self._min_x, self._max_x, self._min_y, self._max_y]
        if None not in axis:
            mpl.axis(axis)

class Roc(PlotBase):
    ''' Handles the plotting of ROC

    Attributes
    ----------

    _semilogx: :obj:`bool`
        If true, X-axis will be semilog10
    '''
    def __init__(self, ctx, scores, test, func_load):
        super(Roc, self).__init__(ctx, scores, test, func_load)
        self._semilogx = None if 'semilogx' not in ctx.meta else\
        ctx.meta['semilogx']
        self._fmr_at = None if 'fmr_at' not in ctx.meta else\
        ctx.meta['fmr_at']
        self._title = 'ROC'
        self._x_label = 'FMR'
        self._y_label = ("1 - FNMR" if self._semilogx else "FNMR")

    def compute(self, idx, dev_neg, dev_pos, dev_fta=None, dev_file=None,
                test_neg=None, test_pos=None, test_fta=None, test_file=None):
        ''' Plot ROC for dev and eval data using
        :py:func:`bob.measure.plot.roc`'''
        mpl.figure(1)
        if self._test:
            linestyle = '-' if not self._split else LINESTYLES[idx % 14]
            plot.roc(
                dev_neg, dev_pos, self._points, self._semilogx,
                color=self._colors[idx], linestyle=linestyle,
                label=self._label('development', dev_file, idx)
            )
            linestyle = '--'
            if self._split:
                mpl.figure(2)
                linestyle = LINESTYLES[idx % 14]

            plot.roc(
                test_neg, test_pos, self._points, self._semilogx,
                color=self._colors[idx], linestyle=linestyle,
                label=self._label('test', test_file, idx)
            )
            if self._fmr_at is not None:
                from .. import farfrr
                test_fmr, test_fnmr = farfrr(test_neg, test_pos, self._fmr_at)
                if self._semilogx:
                    test_fnmr = 1 - test_fnmr
                mpl.scatter(test_fmr, test_fnmr, c=self._colors[idx], s=30)
        else:
            plot.roc(
                dev_neg, dev_pos, self._points, self._semilogx,
                color=self._colors[idx], linestyle=LINESTYLES[idx % 14],
                label=self._label('development', dev_file, idx)
            )

    def end_process(self):
        ''' Draw vertical line on the dev plot at the given fmr and print the
        corresponding points on the test plot for all the systems '''
        #draw vertical lines
        if self._fmr_at is not None:
            mpl.figure(1)
            mpl.plot([self._fmr_at, self._fmr_at], [0., 1.], "--", color='black')
        super(Roc, self).end_process()

class Det(PlotBase):
    ''' Handles the plotting of DET '''
    def __init__(self, ctx, scores, test, func_load):
        super(Det, self).__init__(ctx, scores, test, func_load)
        self._title = 'DET'

    def compute(self, idx, dev_neg, dev_pos, dev_fta=None, dev_file=None,
                test_neg=None, test_pos=None, test_fta=None, test_file=None):
        ''' Plot DET for dev and eval data using
        :py:func:`bob.measure.plot.det`'''
        mpl.figure(1)
        if self._test and test_neg is not None:
            linestyle = '-' if not self._split else LINESTYLES[idx % 14]
            plot.det(
                dev_neg, dev_pos, self._points, color=self._colors[idx],
                linestyle=linestyle, axisfontsize=self._axisfontsize,
                label=self._label('development', dev_file, idx,)
            )
            if self._split:
                mpl.figure(2)
            linestyle = '--' if not self._split else LINESTYLES[idx % 14]
            plot.det(
                test_neg, test_pos, self._points, color=self._colors[idx],
                linestyle=linestyle, axisfontsize=self._axisfontsize,
                label=self._label('test', test_file, idx)
            )
        else:
            plot.det(
                dev_neg, dev_pos, self._points, color=self._colors[idx],
                linestyle=LINESTYLES[idx % 14], axisfontsize=self._axisfontsize,
                label=self._label('development', dev_file, idx)
            )

    def _set_axis(self):
        axis = [self._min_x, self._max_x, self._min_y, self._max_y]
        if None not in axis:
            plot.det_axis(axis)

class Epc(PlotBase):
    ''' Handles the plotting of EPC '''
    def __init__(self, ctx, scores, test, func_load):
        super(Epc, self).__init__(ctx, scores, test, func_load)
        if 'dev-scores' not in self._ctx.meta or 'test-scores' not in self._ctx.meta:
            raise click.UsageError("EPC requires dev and test score files")
        self._title = 'EPC'
        self._x_label = 'Cost'
        self._y_label = 'Min. HTER (%)'
        self._test = True #always test data with EPC

    def compute(self, idx, dev_neg, dev_pos, dev_fta=None, dev_file=None,
                test_neg=None, test_pos=None, test_fta=None, test_file=None):
        ''' Plot EPC using
        :py:func:`bob.measure.plot.epc`'''
        plot.epc(
            dev_neg, dev_pos, test_neg, test_pos, self._points,
            color=self._colors[idx], linestyle=LINESTYLES[idx % 14],
            label=self._label('curve', dev_file + "_" + test_file, idx)
        )

class Hist(PlotBase):
    ''' Handles the plotting of score histograms 

    Attributes
    ----------

    _nbins: :obj:`int`, str
    Number of bins. Default: `auto`

    _thres: :obj:`float`
        If given, this threshold will be used in the plots

    _criter: str
        Criterion to compute threshold (eer or hter)
    '''
    def __init__(self, ctx, scores, test, func_load):
        super(Hist, self).__init__(ctx, scores, test, func_load)
        self._nbins = None if 'nbins' not in ctx.meta else ctx.meta['nbins']
        self._thres = None if 'thres' not in ctx.meta else ctx.meta['thres']
        self._criter = None if 'criter' not in ctx.meta else ctx.meta['criter']
        self._y_label = 'Dev. Scores \n (normalized)' if self._test else \
                'Normalized Count'
        self._x_label = 'Score values' if not self._test else ''
        self._end_setup_plot = False

    def compute(self, idx, dev_neg, dev_pos, dev_fta=None, dev_file=None,
                test_neg=None, test_pos=None, test_fta=None, test_file=None):
        ''' Draw histograms of negative and positive scores.'''
        threshold = utils.get_thres(self._criter, dev_neg, dev_pos) \
                if self._thres is None else self._thres

        fig = mpl.figure()
        if test_neg is not None:
            mpl.subplot(2, 1, 1)
            all_scores = numpy.hstack((dev_neg, test_neg, dev_pos, test_pos))
        else:
            all_scores = numpy.hstack((dev_neg, dev_pos))

        score_range = all_scores.min(), all_scores.max()

        def _setup_hist(neg, pos, xlim, thres, y_label=None):
            mpl.hist(neg, label='Positives', normed=True, color='red',
                     alpha=0.5, bins=self._nbins)
            mpl.hist(pos, label='Negatives', normed=True, color='blue',
                     alpha=0.5, bins=self._nbins)
            mpl.xlim(*xlim)
            _, _, ymax, ymin = mpl.axis()
            mpl.vlines(
                thres, ymin, ymax, color='black',
                label=('' if self._criter is None else self._criter.upper()),
                linestyle='dashed'
            )
            mpl.grid(True, alpha=0.5)
            mpl.ylabel(self._y_label if y_label is None else y_label)
            mpl.xlabel(self._x_label)

        title = dev_file + (" / %s" % test_file if self._test else "")
        mpl.title('Scores  (%s)' % title)
        _setup_hist(dev_neg, dev_pos, score_range, threshold)
        if test_neg is not None:
            ax = mpl.gca()
            ax.axes.get_xaxis().set_ticklabels([])
            mpl.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, -0.01),
                       fontsize=10)
        else:
            mpl.legend(loc='best', fancybox=True, framealpha=0.5)

        if test_neg is not None:
            mpl.subplot(2, 1, 2)
            _setup_hist(
                test_neg, test_pos, score_range, threshold,
                y_label='Test Scores \n (normalized)'
            )
        fig.set_tight_layout(True)
        self._pdf_page.savefig(fig)
