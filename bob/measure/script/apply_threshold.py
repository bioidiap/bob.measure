#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Wed 28 Sep 2016 17:55:17 CEST


"""Applies a threshold to score file and reports error rates

Usage: %(prog)s [-v...] [options] <threshold> <scores>
       %(prog)s --help
       %(prog)s --version


Arguments:
  <threshold>  The threshold value to apply (float)
  <scores>     Path to the file containing the scores where to apply the
               threshold and calculate error rates

Options:
  -h, --help                       Shows this help message and exits
  -V, --version                    Prints the version and exits
  -v, --verbose                    Increases the output verbosity level


Examples:

  Applies the threshold of 0.5 to the scores file in scores.txt and reports:

     $ %(prog)s 0.5 scores.txt

"""


import os
import sys

import logging
__logging_format__='[%(levelname)s] %(message)s'
logging.basicConfig(format=__logging_format__)
logger = logging.getLogger('bob')

from .eval_threshold import apthres


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

  # handles threshold validation
  try:
    args['<threshold>'] = float(args['<threshold>'])
  except:
    raise docopt.DocoptExit("cannot convert %s into float for threshold" % \
        args['<threshold>'])

  from ..load import load_score, get_negatives_positives
  neg, pos = get_negatives_positives(load_score(args['<scores>']))

  apthres(neg, pos, args['<threshold>'])

  return 0
