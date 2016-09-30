#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Wed 28 Sep 2016 15:39:05 CEST

"""Runs error analysis on score sets

  1. Computes the threshold using either EER or min. HTER criteria on
     development set scores
  2. Applies the above threshold on test set scores to compute the HTER
  3. Reports error rates on the console
  4. Plots ROC, EPC, DET curves and score distributions to a multi-page PDF
     file (unless --no-plot is passed)


Usage: %(prog)s [-v...] [options] <dev-scores> <test-scores>
       %(prog)s --help
       %(prog)s --version


Arguments:
  <dev-scores>   Path to the file containing the development scores
  <test-scores>  Path to the file containing the test scores


Options:
  -h, --help                  Shows this help message and exits
  -V, --version               Prints the version and exits
  -v, --verbose               Increases the output verbosity level
  -n <int>, --points=<int>    Number of points to use in the curves
                              [default: 100]
  -o <path>, --output=<path>  Name of the output file that will contain the
                              plots [default: curves.pdf]
  -x, --no-plot               If set, then I'll execute no plotting


Examples:

  1. Specify a different output filename

     $ %(prog)s -vv --output=mycurves.pdf dev.scores test.scores

  2. Specify a different number of points

     $ %(prog)s --points=500 dev.scores test.scores

  3. Don't plot (only calculate thresholds)

     $ %(prog)s --no-plot dev.scores test.scores

"""

import os
import sys
import numpy

import logging
__logging_format__='[%(levelname)s] %(message)s'
logging.basicConfig(format=__logging_format__)
logger = logging.getLogger('bob')


def print_crit(dev_neg, dev_pos, test_neg, test_pos, crit):
  """Prints a single output line that contains all info for a given criterion"""

  if crit == 'EER':
    from .. import eer_threshold
    thres = eer_threshold(dev_neg, dev_pos)
  else:
    from .. import min_hter_threshold
    thres = min_hter_threshold(dev_neg, dev_pos)

  from .. import farfrr
  dev_far, dev_frr = farfrr(dev_neg, dev_pos, thres)
  dev_hter = (dev_far + dev_frr)/2.0

  test_far, test_frr = farfrr(test_neg, test_pos, thres)
  test_hter = (test_far + test_frr)/2.0

  print("[Min. criterion: %s] Threshold on Development set: %e" % (crit, thres))

  dev_ni = dev_neg.shape[0] #number of impostors
  dev_fa = int(round(dev_far*dev_ni)) #number of false accepts
  dev_nc = dev_pos.shape[0] #number of clients
  dev_fr = int(round(dev_frr*dev_nc)) #number of false rejects
  test_ni = test_neg.shape[0] #number of impostors
  test_fa = int(round(test_far*test_ni)) #number of false accepts
  test_nc = test_pos.shape[0] #number of clients
  test_fr = int(round(test_frr*test_nc)) #number of false rejects

  dev_far_str = "%.3f%% (%d/%d)" % (100*dev_far, dev_fa, dev_ni)
  test_far_str = "%.3f%% (%d/%d)" % (100*test_far, test_fa, test_ni)
  dev_frr_str = "%.3f%% (%d/%d)" % (100*dev_frr, dev_fr, dev_nc)
  test_frr_str = "%.3f%% (%d/%d)" % (100*test_frr, test_fr, test_nc)
  dev_max_len = max(len(dev_far_str), len(dev_frr_str))
  test_max_len = max(len(test_far_str), len(test_frr_str))

  def fmt(s, space):
    return ('%' + ('%d' % space) + 's') % s

  print("       | %s | %s" % (fmt("Development", -1*dev_max_len),
    fmt("Test", -1*test_max_len)))
  print("-------+-%s-+-%s" % (dev_max_len*"-", (2+test_max_len)*"-"))
  print("  FAR  | %s | %s" % (fmt(dev_far_str, dev_max_len), fmt(test_far_str,
    test_max_len)))
  print("  FRR  | %s | %s" % (fmt(dev_frr_str, dev_max_len), fmt(test_frr_str,
    test_max_len)))
  dev_hter_str = "%.3f%%" % (100*dev_hter)
  test_hter_str = "%.3f%%" % (100*test_hter)
  print("  HTER | %s | %s" % (fmt(dev_hter_str, -1*dev_max_len),
    fmt(test_hter_str, -1*test_max_len)))


def plots(dev_neg, dev_pos, test_neg, test_pos, crit, points, filename):
  """Saves ROC, DET and EPC curves on the file pointed out by filename."""

  from .. import plot

  import matplotlib
  if not hasattr(matplotlib, 'backends'): matplotlib.use('pdf')
  import matplotlib.pyplot as mpl
  from matplotlib.backends.backend_pdf import PdfPages

  pp = PdfPages(filename)

  # ROC
  fig = mpl.figure()
  plot.roc(dev_neg, dev_pos, points, color=(0.3,0.3,0.3),
      linestyle='--', dashes=(6,2), label='development')
  plot.roc(test_neg, test_pos, points, color=(0,0,0),
      linestyle='-', label='test')
  mpl.axis([0,40,0,40])
  mpl.title("ROC Curve")
  mpl.xlabel('FAR (%)')
  mpl.ylabel('FRR (%)')
  mpl.grid(True, color=(0.3,0.3,0.3))
  mpl.legend()
  pp.savefig(fig)

  # DET
  fig = mpl.figure()
  plot.det(dev_neg, dev_pos, points, color=(0.3,0.3,0.3),
      linestyle='--', dashes=(6,2), label='development')
  plot.det(test_neg, test_pos, points, color=(0,0,0),
      linestyle='-', label='test')
  plot.det_axis([0.01, 40, 0.01, 40])
  mpl.title("DET Curve")
  mpl.xlabel('FAR (%)')
  mpl.ylabel('FRR (%)')
  mpl.grid(True, color=(0.3,0.3,0.3))
  mpl.legend()
  pp.savefig(fig)

  # EPC
  fig = mpl.figure()
  plot.epc(dev_neg, dev_pos, test_neg, test_pos, points,
      color=(0,0,0), linestyle='-')
  mpl.title('EPC Curve')
  mpl.xlabel('Cost')
  mpl.ylabel('Min. HTER (%)')
  mpl.grid(True, color=(0.3,0.3,0.3))
  pp.savefig(fig)

  # Distribution for dev and test scores on the same page
  if crit == 'EER':
    from .. import eer_threshold
    thres = eer_threshold(dev_neg, dev_pos)
  else:
    from .. import min_hter_threshold
    thres = min_hter_threshold(dev_neg, dev_pos)

  mpl.subplot(2,1,1)
  nbins=20
  all_scores = numpy.hstack((dev_neg, test_neg, dev_pos, test_pos))
  score_range = all_scores.min(), all_scores.max()
  mpl.hist(dev_neg, label='Impostors', normed=True, color='red', alpha=0.5,
      bins=nbins)
  mpl.hist(dev_pos, label='Genuine', normed=True, color='blue', alpha=0.5,
      bins=nbins)
  mpl.xlim(*score_range)
  _, _, ymax, ymin = mpl.axis()
  mpl.vlines(thres, ymin, ymax, color='black', label='EER', linestyle='dashed')
  mpl.ylabel('Dev. Scores (normalized)')
  ax = mpl.gca()
  ax.axes.get_xaxis().set_ticklabels([])
  mpl.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, -0.01),
      fontsize=10)
  mpl.title('Score Distributions')
  mpl.grid(True, alpha=0.5)

  mpl.subplot(2,1,2)
  mpl.hist(test_neg, label='Impostors', normed=True, color='red', alpha=0.5,
      bins=nbins)
  mpl.hist(test_pos, label='Genuine', normed=True, color='blue', alpha=0.5,
      bins=nbins)
  mpl.ylabel('Test Scores (normalized)')
  mpl.xlabel('Score value')
  mpl.xlim(*score_range)
  _, _, ymax, ymin = mpl.axis()
  mpl.vlines(thres, ymin, ymax, color='black', label='EER', linestyle='dashed')
  mpl.grid(True, alpha=0.5)
  pp.savefig(fig)

  pp.close()


def main(user_input=None):

  if user_input is not None:
    argv = user_input
  else:
    argv = sys.argv[1:]

  import docopt
  import pkg_resources

  completions = dict(
      prog=os.path.basename(sys.argv[0]),
      version=pkg_resources.require('bob.measure')[0].version
      )

  args = docopt.docopt(
      __doc__ % completions,
      argv=argv,
      version=completions['version'],
      )

  # Sets-up logging
  if args['--verbose'] == 1: logging.getLogger().setLevel(logging.INFO)
  elif args['--verbose'] >= 2: logging.getLogger().setLevel(logging.DEBUG)

  # Checks number of points option
  try:
    args['--points'] = int(args['--points'])
  except:
    raise docopt.DocoptExit("cannot convert %s into int for points" % \
        args['--points'])

  if args['--points'] <= 0:
    raise docopt.DocoptExit('Number of points (--points) should greater ' \
        'than zero')

  from ..load import load_score, get_negatives_positives
  dev_neg, dev_pos = get_negatives_positives(load_score(args['<dev-scores>']))
  test_neg, test_pos = get_negatives_positives(load_score(args['<test-scores>']))

  print_crit(dev_neg, dev_pos, test_neg, test_pos, 'EER')
  print_crit(dev_neg, dev_pos, test_neg, test_pos, 'Min. HTER')

  if not args['--no-plot']:
    plots(dev_neg, dev_pos, test_neg, test_pos, 'EER', args['--points'],
        args['--output'])
    print("[Plots] Performance curves => '%s'" % args['--output'])

  return 0
