#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Wed 28 Sep 2016 16:56:52 CEST


"""Computes the threshold following a minimization criteria on input scores

Usage: %(prog)s [-v...] [options] <scores>
       %(prog)s --help
       %(prog)s --version


Arguments:
  <scores>  Path to the file containing the scores to be used for calculating
            the threshold


Options:
  -h, --help                     Shows this help message and exits
  -V, --version                  Prints the version and exits
  -v, --verbose                  Increases the output verbosity level
  -c <crit>, --criterion=<crit>  The minimization criterion to use (choose
                                 between mhter, mwer or eer) [default: eer]
  -w <float>, --cost=<float>     The value w of the cost when minimizing using
                                 the minimum weighter error rate (mwer)
                                 criterion. This value is ignored for eer or
                                 mhter criteria. [default: 0.5]


Examples:

  1. Specify a different criteria (only mhter, mwer or eer accepted):

     $ %(prog)s --criterion=mhter scores.txt

  2. Calculate the threshold that minimizes the weither HTER for a cost of 0.4:

    $ %(prog)s --criterion=mwer --cost=0.4 scores.txt

  3. Parse your input using a 5-column format

    $ %(prog)s scores.txt

"""


import os
import sys

import logging
__logging_format__='[%(levelname)s] %(message)s'
logging.basicConfig(format=__logging_format__)
logger = logging.getLogger('bob')


def apthres(neg, pos, thres):
  """Prints a single output line that contains all info for the threshold"""

  from .. import farfrr

  far, frr = farfrr(neg, pos, thres)
  hter = (far + frr)/2.0

  ni = neg.shape[0] #number of impostors
  fa = int(round(far*ni)) #number of false accepts
  nc = pos.shape[0] #number of clients
  fr = int(round(frr*nc)) #number of false rejects

  print("FAR : %.3f%% (%d/%d)" % (100*far, fa, ni))
  print("FRR : %.3f%% (%d/%d)" % (100*frr, fr, nc))
  print("HTER: %.3f%%" % (100*hter,))


def calculate(neg, pos, crit, cost):
  """Returns the threshold given a certain criteria"""

  if crit == 'eer':
    from .. import eer_threshold
    return eer_threshold(neg, pos)
  elif crit == 'mhter':
    from .. import min_hter_threshold
    return min_hter_threshold(neg, pos)

  # defaults to the minimum of the weighter error rate
  from .. import min_weighted_error_rate_threshold
  return min_weighted_error_rate_threshold(neg, pos, cost)


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

  # validates criterion
  valid_criteria = ('eer', 'mhter', 'mwer')
  if args['--criterion'] not in valid_criteria:
    raise docopt.DocoptExit("--criterion must be one of %s" % \
        ', '.join(valid_criteria))

  # handles cost validation
  try:
    args['--cost'] = float(args['--cost'])
  except:
    raise docopt.DocoptExit("cannot convert %s into float for cost" % \
        args['--cost'])

  if args['--cost'] < 0.0 or args['--cost'] > 1.0:
    docopt.DocoptExit("cost should lie between 0.0 and 1.0")

  from ..load import load_score, get_negatives_positives
  neg, pos = get_negatives_positives(load_score(args['<scores>']))

  t = calculate(neg, pos, args['--criterion'], args['--cost'])
  print("Threshold:", t)
  apthres(neg, pos, t)

  return 0
