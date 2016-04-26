#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <manuel.guenther@idiap.ch>
# Tue Jan  8 13:36:12 CET 2013
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

from __future__ import print_function

"""This script computes and plot a cumulative rank characteristics (CMC) curve
from a score file in four or five column format.

Note: The score file has to contain the exact probe file names as the 3rd
(4column) or 4th (5column) column.
"""

import os
import sys

def parse_command_line(command_line_options):
  """Parse the program options"""

  usage = 'usage: %s [arguments]' % os.path.basename(sys.argv[0])

  import argparse
  parser = argparse.ArgumentParser(usage=usage, description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # This option is not normally shown to the user...
  parser.add_argument('--self-test', action = 'store_true', help = argparse.SUPPRESS)
  parser.add_argument('-s', '--score-file', required = True, help = 'The score file in 4 or 5 column format to test.')
  parser.add_argument('-o', '--output-pdf-file', default = 'cmc.pdf', help = 'The PDF file to write.')
  parser.add_argument('-l', '--log-x-scale', action='store_true', help = 'Plot logarithmic Rank axis.')
  parser.add_argument('-r', '--rank', type=int, help = 'Plot Detection & Identification rate curve for the given rank instead of the CMC curve.')
  parser.add_argument('-x', '--no-plot', action = 'store_true', help = 'Do not print a PDF file, but only report the results.')
  parser.add_argument('-p', '--parser', default = '4column', choices = ('4column', '5column'), help = 'The type of the score file.')

  args = parser.parse_args(command_line_options)

  if args.self_test:
    # then we go into test mode, all input is preset
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="bobtest_")
    args.output_pdf_file = os.path.join(temp_dir, "cmc.pdf")
    print("temporary using file", args.output_pdf_file)

  return args

def main(command_line_options = None):
  """Computes and plots the CMC curve."""

  from .. import load, plot, recognition_rate

  args = parse_command_line(command_line_options)

  # read data
  if not os.path.isfile(args.score_file): raise IOError("The given score file does not exist")
  # pythonic way: create inline dictionary "{...}", index with desired value "[...]", execute function "(...)"
  data = {'4column' : load.cmc_four_column, '5column' : load.cmc_five_column}[args.parser](args.score_file)

  # compute recognition rate
  rr = recognition_rate(data, args.rank)
  print("Recognition rate for score file", args.score_file, "is %3.2f%%" % (rr * 100))

  if not args.no_plot:
    # compute CMC
    import matplotlib
    if not hasattr(matplotlib, 'backends'): matplotlib.use('pdf')
    import matplotlib.pyplot as mpl
    from matplotlib.backends.backend_pdf import PdfPages

    pp = PdfPages(args.output_pdf_file)

    # CMC
    fig = mpl.figure()
    if args.rank is None:
      max_rank = plot.cmc(data, color=(0,0,1), linestyle='--', dashes=(6,2), logx = args.log_x_scale)
      mpl.title("CMC Curve")
      if args.log_x_scale:
        mpl.xlabel('Rank (log)')
      else:
        mpl.xlabel('Rank')
      mpl.ylabel('Recognition Rate in %')

      ticks = [int(t) for t in mpl.xticks()[0]]
      mpl.xticks(ticks, ticks)
      mpl.xlim([1, max_rank])
    else:
      plot.detection_identification_curve(data, rank = args.rank, color=(0,0,1), linestyle='--', dashes=(6,2), logx = args.log_x_scale)
      mpl.title("Detection & Identification Curve")
      if args.log_x_scale:
        mpl.xlabel('False Acceptance Rate (log) in %')
      else:
        mpl.xlabel('False Acceptance Rate in %')
      mpl.ylabel('Detection & Identification Rate in %')

      ticks = ["%s"%(t*100) for t in mpl.xticks()[0]]
      mpl.xticks(mpl.xticks()[0], ticks)
      mpl.xlim([1e-4, 1])

    mpl.grid(True, color=(0.3,0.3,0.3))
    mpl.ylim(ymax=101)
    # convert log-scale ticks to normal numbers

    pp.savefig(fig)
    pp.close()

  if args.self_test: #remove output file + tmp directory
    import shutil
    shutil.rmtree(os.path.dirname(args.output_pdf_file))

  return 0

if __name__ == '__main__':
  main(sys.argv[1:])
