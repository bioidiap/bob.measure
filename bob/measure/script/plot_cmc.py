#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Wed 28 Sep 2016 21:24:46 CEST

"""Computes and plots a cumulative rank characteristics (CMC) curve

Usage: %(prog)s [-v...] [options] <scores>
       %(prog)s --help
       %(prog)s --version

Arguments:

  <scores>  The score file in 4 or 5 column format to test


Options:

  -h, --help                  Shows this help message and exits
  -V, --version               Prints the version and exits
  -v, --verbose               Increases the output verbosity level
  -o <path>, --output=<path>  Name of the output file that will contain the
                              plots [default: cmc.pdf]
  -x, --no-plot               If set, then I'll execute no plotting
  -l, --log-x-scale           If set, plots logarithmic rank axis
  -r <int>, --rank=<int>      Plot detection & identification rate curve for
                              the given rank instead of the CMC curve.

"""

from __future__ import print_function

import os
import sys


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

  # Validates rank
  if args['--rank'] is not None:
    try:
      args['--rank'] = int(args['--rank'])
    except:
      raise docopt.DocoptExit("cannot convert %s into int for rank" % \
          args['--rank'])

    if args['--rank'] <= 0:
      raise docopt.DocoptExit('Rank (--rank) should greater than zero')

  from .. import load

  # Loads score file
  f = load.open_file(args['<scores>'])
  try:
    line = f.readline()
    ncolumns = len(line.split())
  except Exception:
    logger.warn('Could not guess the number of columns in file: {}. '
                'Assuming 4 column format.'.format(args['<scores>']))
    ncolumns = 4
  finally:
    f.close()

  if ncolumns == 4:
    data = load.cmc_four_column(args['<scores>'])
  else:
    data = load.cmc_five_column(args['<scores>'])

  # compute recognition rate
  from .. import recognition_rate
  rr = recognition_rate(data, args['--rank'])
  print("Recognition rate for score file %s is %3.2f%%" % (args['<scores>'],
    rr * 100))

  if not args['--no-plot']:

    from .. import plot

    # compute CMC
    import matplotlib
    if not hasattr(matplotlib, 'backends'): matplotlib.use('pdf')
    import matplotlib.pyplot as mpl
    from matplotlib.backends.backend_pdf import PdfPages

    pp = PdfPages(args['--output'])

    # CMC
    fig = mpl.figure()
    if args['--rank'] is None:
      max_rank = plot.cmc(data, color=(0,0,1), linestyle='--', dashes=(6,2),
          logx = args['--log-x-scale'])
      mpl.title("CMC Curve")
      if args['--log-x-scale']:
        mpl.xlabel('Rank (log)')
      else:
        mpl.xlabel('Rank')
      mpl.ylabel('Recognition Rate in %')

      ticks = [int(t) for t in mpl.xticks()[0]]
      mpl.xticks(ticks, ticks)
      mpl.xlim([1, max_rank])

    else:
      plot.detection_identification_curve(data, rank = args['--rank'],
          color=(0,0,1), linestyle='--', dashes=(6,2),
          logx = args['--log-x-scale'])
      mpl.title("Detection \& Identification Curve")
      if args['--log-x-scale']:
        mpl.xlabel('False Acceptance Rate (log) in %')
      else:
        mpl.xlabel('False Acceptance Rate in %')
      mpl.ylabel('Detection \& Identification Rate in %')

      ticks = ["%s"%(t*100) for t in mpl.xticks()[0]]
      mpl.xticks(mpl.xticks()[0], ticks)
      mpl.xlim([1e-4, 1])

    mpl.grid(True, color=(0.3,0.3,0.3))
    mpl.ylim(ymax=101)
    # convert log-scale ticks to normal numbers

    pp.savefig(fig)
    pp.close()

  return 0
