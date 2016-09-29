#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tue Jul 2 14:52:49 CEST 2013

"""Collects results of verification experiments recursively, reports results

This script parses through the given directory, collects all results of
verification experiments that are stored in file with the given file name.  It
supports the split into development and test set of the data, as well as
ZT-normalized scores.

All result files are parsed and evaluated. For each directory, the following
information are given in columns:

  * The Equal Error Rate of the development set
  * The Equal Error Rate of the development set after ZT-Normalization
  * The Half Total Error Rate of the evaluation set
  * The Half Total Error Rate of the evaluation set after ZT-Normalization
  * The sub-directory where the scores can be found

The measure type of the development set can be changed to compute "HTER" or
"FAR" thresholds instead, using the --criterion option.

Usage: %(prog)s [-v...] [options]
       %(prog)s --help
       %(prog)s --version


Options:
  -h, --help                      Shows this help message and exit
  -V, --version                   Prints the version and exits
  -v, --verbose                   Increases the output verbosity level
  -d <path>, --devel-name=<path>  Name of the file containing the development
                                  scores [default: scores-dev]
  -e <path>, --eval-name=<path>   Name of the file containing the evaluation
                                  scores [default: scores-eval]
  -D <dir>, --directory=<dir>     The directory where the results should be
                                  collected from [default: .]
  -n <dir>, --nonorm-dir=<dir>    Directory where the unnormalized scores are
                                  found [default: nonorm]
  -z <dir>, --ztnorm-dir=<dir>    Directory where the normalized scores are
                                  found [default: ztnorm]
  -s, --sort                      If set, sorts the results
  -k <key>, --sort-key=<key>      Sorts the results according to the given key.
                                  May be one of "nonorm_dev", "nonorm_eval",
                                  "ztnorm_dev", "ztnorm_eval" or "dir"
                                  [default: dir]
  -c <crit>, --criterion=<crit>   Report Equal Rates (EER) rather than Half
                                  Total Error Rate (HTER). Choose between
                                  "HTER", "EER" or "FAR" [default: HTER]
  -o <path>, --output=<path>      If set, outputs results to a file named after
                                  the option. If not set, writes to the console

"""

import os
import sys

import logging
__logging_format__='[%(levelname)s] %(message)s'
logging.basicConfig(format=__logging_format__)
logger = logging.getLogger('bob')


class Result:
  def __init__(self, dir, args):
    self.dir = dir
    self.args = args
    self.nonorm_dev = None
    self.nonorm_eval = None
    self.ztnorm_dev = None
    self.ztnorm_eval = None

  def __calculate__(self, dev_file, eval_file = None):
    from ..load import load_score, get_negatives_positives
    from .. import eer_threshold, min_hter_threshold, far_threshold, farfrr

    dev_neg, dev_pos = get_negatives_positives(load_score(dev_file))

    # switch which threshold function to use;
    # THIS f***ing piece of code really is what python authors propose:
    threshold = {
      'EER'  : eer_threshold,
      'HTER' : min_hter_threshold,
      'FAR'  : far_threshold
    } [self.args['--criterion']](dev_neg, dev_pos)

    # compute far and frr for the given threshold
    dev_far, dev_frr = farfrr(dev_neg, dev_pos, threshold)
    dev_hter = (dev_far + dev_frr)/2.0

    if eval_file:
      eval_neg, eval_pos = get_negatives_positives(load_score(eval_file))
      eval_far, eval_frr = farfrr(eval_neg, eval_pos, threshold)
      eval_hter = (eval_far + eval_frr)/2.0
    else:
      eval_hter = None

    if self.args['--criterion'] == 'FAR':
      return (dev_frr, eval_frr)
    else:
      return (dev_hter, eval_hter)

  def nonorm(self, dev_file, eval_file = None):
    (self.nonorm_dev, self.nonorm_eval) = \
        self.__calculate__(dev_file, eval_file)

  def ztnorm(self, dev_file, eval_file = None):
    (self.ztnorm_dev, self.ztnorm_eval) = \
        self.__calculate__(dev_file, eval_file)

  def __str__(self):
    str = ""
    for v in [self.nonorm_dev, self.ztnorm_dev, self.nonorm_eval, self.ztnorm_eval]:
      if v:
        val = "% 2.3f%%"%(v*100)
      else:
        val = "None"
      cnt = 16-len(val)
      str += " "*cnt + val
    str += "        %s"%self.dir
    return str[5:]


results = []


def add_results(args, nonorm, ztnorm = None):

  r = Result(os.path.dirname(nonorm).replace(os.getcwd()+"/", ""), args)
  print("Adding results from directory", r.dir)
  # check if the results files are there
  dev_file = os.path.join(nonorm, args['--devel-name'])
  eval_file = os.path.join(nonorm, args['--eval-name'])
  if os.path.isfile(dev_file):
    if os.path.isfile(eval_file):
      r.nonorm(dev_file, eval_file)
    else:
      r.nonorm(dev_file)

  if ztnorm:
    dev_file = os.path.join(ztnorm, args['--devel-name'])
    eval_file = os.path.join(ztnorm, args['--eval-name'])
    if os.path.isfile(dev_file):
      if os.path.isfile(eval_file):
        r.ztnorm(dev_file, eval_file)
      else:
        r.ztnorm(dev_file)

  results.append(r)


def recurse(args, path):
  dir_list = os.listdir(path)

  # check if the score directories are included in the current path
  if args['--nonorm-dir'] in dir_list:
    if args['--ztnorm-dir'] in dir_list:
      add_results(args, os.path.join(path, args['--nonorm-dir']),
          os.path.join(path, args['--ztnorm-dir']))
    else:
      add_results(args, os.path.join(path, args['--nonorm-dir']))

  for e in dir_list:
    real_path = os.path.join(path, e)
    if os.path.isdir(real_path):
      recurse(args, real_path)


def table():
  A = " "*2 + 'dev  nonorm'+ " "*5 + 'dev  ztnorm' + " "*6 + 'eval nonorm' + " "*4 + 'eval ztnorm' + " "*12 + 'directory\n'
  A += "-"*100+"\n"
  for r in results:
    A += str(r) + "\n"
  return A


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

  # checks sort-key
  valid_sort_keys = 'nonorm_dev nonorm_eval ztnorm_dev ztnorm_eval dir'.split()
  if args['--sort-key'] not in valid_sort_keys:
    raise docopt.DocoptExit('--sort-key must be one of %s' % \
        ', '.join(valid_sort_keys))

  # checks criterion
  valid_criterion = 'HTER EER FAR'.split()
  if args['--criterion'] not in valid_criterion:
    raise docopt.DocoptExit('--criterion must be one of %s' % \
        ', '.join(valid_criterion))

  recurse(args, args['--directory'])

  if args['--sort']:
    import operator
    results.sort(key=operator.attrgetter(args['--sort-key']))

  if args['--output']:
    f = open(args['--output'], "w")
    f.writelines(table())
    f.close()
  else:
    print(table())
