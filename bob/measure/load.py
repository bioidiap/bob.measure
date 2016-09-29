#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Mon 23 May 2011 16:23:05 CEST

"""A set of utilities to load score files with different formats.
"""

import numpy
import tarfile
import os

import logging
logger = logging.getLogger('bob.measure')


def open_file(filename, mode='rt'):
  """Opens the given score file for reading.

  Score files might be raw text files, or a tar-file including a single score
  file inside.


  Parameters:

    filename (:py:class:`str`, ``file-like``): The name of the score file to
      open, or a file-like object open for reading. If a file name is given,
      the according file might be a raw text file or a (compressed) tar file
      containing a raw text file.


  Returns:


    ``file-like``: A read-only file-like object as it would be returned by
    :py:func:`open`.

  """

  if not isinstance(filename, str) and hasattr(filename, 'read'):
    # It seems that this is an open file
    return filename

  if not os.path.isfile(filename):
    raise IOError("Score file '%s' does not exist." % filename)
  if not tarfile.is_tarfile(filename):
    return open(filename, mode)

  # open the tar file for reading
  tar = tarfile.open(filename, 'r')
  # get the first file in the tar file
  tar_info = tar.next()
  while tar_info is not None and not tar_info.isfile():
    tar_info = tar.next()
  # check that one file was found in the archive
  if tar_info is None:
    raise IOError("The given file is a .tar file, but it does not contain any file.")

  # open the file for reading
  return tar.extractfile(tar_info)


def four_column(filename):
  """Loads a score set from a single file and yield its lines

  Loads a score set from a single file and yield its lines (to avoid loading
  the score file at once into memory).  This function verifies that all fields
  are correctly placed and contain valid fields.  The score file must contain
  the following information in each line:

  .. code-block:: text

     claimed_id real_id test_label score


  Parameters:

    filename (:py:class:`str`, ``file-like``): The file object that will be
      opened with :py:func:`open_file` containing the scores.


  Returns:

    str: The claimed identity -- the client name of the model that was used in
    the comparison

    str: The real identity -- the client name of the probe that was used in the
    comparison

    str: A label of the probe -- usually the probe file name, or the probe id

    float: The result of the comparison of the model and the probe

  """

  for i, l in enumerate(open_file(filename)):
    if isinstance(l, bytes): l = l.decode('utf-8')
    s = l.strip()
    if len(s) == 0 or s[0] == '#': continue #empty or comment
    field = [k.strip() for k in s.split()]
    if len(field) < 4:
      raise SyntaxError('Line %d of file "%s" is invalid: %s' % (i, filename, l))
    try:
      score = float(field[3])
    except:
      raise SyntaxError('Cannot convert score to float at line %d of file "%s": %s' % (i, filename, l))
    yield (field[0], field[1], field[2], score)


def split_four_column(filename):
  """Loads a score set from a single file and splits the scores

  Loads a score set from a single file and splits the scores between negatives
  and positives. The score file has to respect the 4 column format as defined
  in the method :py:func:`four_column`.

  This method avoids loading and allocating memory for the strings present in
  the file. We only keep the scores.


  Parameters:

    filename (:py:class:`str`, ``file-like``): The file object that will be
      opened with :py:func:`open_file` containing the scores.


  Returns:

    negatives (array): 1D float array containing the list of scores, for which
    the ``claimed_id`` and the ``real_id`` differed (see
    :py:func:`four_column`)

    positivies (array): 1D float array containing the list of scores, for which
    the ``claimed_id`` and the ``real_id`` are identical (see
    :py:func:`four_column`)

  """

  score_lines = load_score(filename, 4)
  return get_negatives_positives(score_lines)


def cmc_four_column(filename):
  """Loads scores to compute CMC curves from a file in four column format.

  The four column file needs to be in the same format as described in
  :py:func:`four_column`, and the ``test_label`` (column 3) has to contain the
  test/probe file name or a probe id.

  This function returns a list of tuples.  For each probe file, the tuple
  consists of a list of negative scores and a list of positive scores.
  Usually, the list of positive scores should contain only one element, but
  more are allowed.  The result of this function can directly be passed to,
  e.g., the :py:func:`bob.measure.cmc` function.


  Parameters:

    filename (:py:class:`str`, ``file-like``): The file object that will be
      opened with :py:func:`open_file` containing the scores.


  Returns:

    list: A list of tuples, where each tuple contains the ``negative`` and
    ``positive`` scores for one probe of the database. Both ``negatives`` and
    ``positives`` can be either an 1D :py:class:`numpy.ndarray` of type
    ``float``, or ``None``.

  """

  # extract positives and negatives
  pos_dict = {}
  neg_dict = {}
  # read four column list
  for (client_id, probe_id, probe_name, score) in four_column(filename):
    # check in which dict we have to put the score
    correct_dict = pos_dict if client_id == probe_id else neg_dict

    # append score
    if probe_name in correct_dict:
      correct_dict[probe_name].append(score)
    else:
      correct_dict[probe_name] = [score]

  # convert that into the desired format
  return _convert_cmc_scores(neg_dict, pos_dict)


def five_column(filename):
  """Loads a score set from a single file and yield its lines

  Loads a score set from a single file and yield its lines (to avoid loading
  the score file at once into memory).  This function verifies that all fields
  are correctly placed and contain valid fields.  The score file must contain
  the following information in each line:

  .. code-block:: text

     claimed_id model_label real_id test_label score


  Parameters:

    filename (:py:class:`str`, ``file-like``): The file object that will be
      opened with :py:func:`open_file` containing the scores.


  Returns:

    str: The claimed identity -- the client name of the model that was used in
    the comparison

    str: A label for the model -- usually the model file name, or the model id

    str: The real identity -- the client name of the probe that was used in the
    comparison

    str: A label of the probe -- usually the probe file name, or the probe id

    float: The result of the comparison of the model and the probe

  """

  for i, l in enumerate(open_file(filename)):
    if isinstance(l, bytes): l = l.decode('utf-8')
    s = l.strip()
    if len(s) == 0 or s[0] == '#': continue #empty or comment
    field = [k.strip() for k in s.split()]
    if len(field) < 5:
      raise SyntaxError('Line %d of file "%s" is invalid: %s' % (i, filename, l))
    try:
      score = float(field[4])
    except:
      raise SyntaxError('Cannot convert score to float at line %d of file "%s": %s' % (i, filename, l))
    yield (field[0], field[1], field[2], field[3], score)


def split_five_column(filename):
  """Loads a score set from a single file and splits the scores

  Loads a score set from a single file in five column format and splits the
  scores between negatives and positives. The score file has to respect the 5
  column format as defined in the method :py:func:`five_column`.

  This method avoids loading and allocating memory for the strings present in
  the file. We only keep the scores.


  Parameters:

    filename (:py:class:`str`, ``file-like``): The file object that will be
      opened with :py:func:`open_file` containing the scores.


  Returns:

    negatives (array): 1D float array containing the list of scores, for which
    the ``claimed_id`` and the ``real_id`` differed (see
    :py:func:`four_column`)

    positivies (array): 1D float array containing the list of scores, for which
    the ``claimed_id`` and the ``real_id`` are identical (see
    :py:func:`four_column`)

  """

  score_lines = load_score(filename, 5)
  return get_negatives_positives(score_lines)


def cmc_five_column(filename):
  """Loads scores to compute CMC curves from a file in five column format.

  The five column file needs to be in the same format as described in
  :py:func:`five_column`, and the ``test_label`` (column 4) has to contain the
  test/probe file name or a probe id.

  This function returns a list of tuples.  For each probe file, the tuple
  consists of a list of negative scores and a list of positive scores.
  Usually, the list of positive scores should contain only one element, but
  more are allowed.  The result of this function can directly be passed to,
  e.g., the :py:func:`bob.measure.cmc` function.


  Parameters:

    filename (:py:class:`str`, ``file-like``): The file object that will be
      opened with :py:func:`open_file` containing the scores.


  Returns:

    list: A list of tuples, where each tuple contains the ``negative`` and
    ``positive`` scores for one probe of the database.

  """
  # extract positives and negatives
  pos_dict = {}
  neg_dict = {}
  # read four column list
  for (client_id, _, probe_id, probe_name, score) in five_column(filename):
    # check in which dict we have to put the score
    correct_dict = pos_dict if client_id == probe_id else neg_dict

    # append score
    if probe_name in correct_dict:
      correct_dict[probe_name].append(score)
    else:
      correct_dict[probe_name] = [score]

  # convert that into the desired format
  return _convert_cmc_scores(neg_dict, pos_dict)


def load_score(filename, ncolumns=None):
  """Load scores using numpy.loadtxt and return the data as a numpy array.

  Parameters:

    filename (:py:class:`str`, ``file-like``): The file object that will be
      opened with :py:func:`open_file` containing the scores.

    ncolumns (:py:class:`int`, optional): 4, 5 or None (the default),
      specifying the number of columns in the score file. If None is provided,
      the number of columns will be guessed.


  Returns:

    array: An array which contains not only the actual scores but also the
    ``claimed_id``, ``real_id``, ``test_label`` and ``['model_label']``

  """

  def convertfunc(x):
    return x

  if ncolumns not in (4, 5):
    f = open_file(filename)
    try:
      line = f.readline()
      ncolumns = len(line.split())
    except Exception:
      logger.warn('Could not guess the number of columns in file: {}. '
                  'Assuming 4 column format.'.format(filename))
      ncolumns = 4
    finally:
      f.close()

  if ncolumns == 4:
    names = ('claimed_id', 'real_id', 'test_label', 'score')
    converters = {
      0: convertfunc,
      1: convertfunc,
      2: convertfunc,
      3: float}

  elif ncolumns == 5:
    names = ('claimed_id', 'model_label', 'real_id', 'test_label', 'score')
    converters = {
      0: convertfunc,
      1: convertfunc,
      2: convertfunc,
      3: convertfunc,
      4: float}
  else:
    raise ValueError("ncolumns of 4 and 5 are supported only.")

  score_lines = numpy.genfromtxt(
    open_file(filename, mode='rb'), dtype=None, names=names,
    converters=converters, invalid_raise=True)
  new_dtype = []
  for name in score_lines.dtype.names[:-1]:
    new_dtype.append((name, str(score_lines.dtype[name]).replace('S', 'U')))
  new_dtype.append(('score', float))
  score_lines = numpy.array(score_lines, new_dtype)
  return score_lines


def get_negatives_positives(score_lines):
  """Take the output of load_score and return negatives and positives.  This
  function aims to replace split_four_column and split_five_column but takes a
  different input. It's up to you to use which one.
  """

  pos_mask = score_lines['claimed_id'] == score_lines['real_id']
  positives = score_lines['score'][pos_mask]
  negatives = score_lines['score'][numpy.logical_not(pos_mask)]
  return (negatives, positives)


def get_negatives_positives_all(score_lines_list):
  """Take a list of outputs of load_score and return stacked negatives and
  positives.
  """

  negatives, positives = [], []
  for score_lines in score_lines_list:
    neg_pos = get_negatives_positives(score_lines)
    negatives.append(neg_pos[0])
    positives.append(neg_pos[1])
  negatives = numpy.vstack(negatives).T
  positives = numpy.vstack(positives).T
  return (negatives, positives)


def get_all_scores(score_lines_list):
  """Take a list of outputs of load_score and return stacked scores"""

  return numpy.vstack([score_lines['score']
                       for score_lines in score_lines_list]).T


def dump_score(filename, score_lines):
  """Dump scores that were loaded using :py:func:`load_score`
  The number of columns is automatically detected.
  """

  if len(score_lines.dtype) == 5:
    fmt = '%s %s %s %s %.9f'
  elif len(score_lines.dtype) == 4:
    fmt = '%s %s %s %.9f'
  else:
    raise ValueError("Only scores with 4 and 5 columns are supported.")
  numpy.savetxt(filename, score_lines, fmt=fmt)


def _convert_cmc_scores(neg_dict, pos_dict):
  """Converts the negative and positive scores read with
  :py:func:`cmc_four_column` or :py:func:`cmc_four_column` into a format that
  is handled by the :py:func:`bob.measure.cmc` and similar functions.
  """

  # convert to lists of tuples of ndarrays (or None)
  probe_names = sorted(set(neg_dict.keys()).union(set(pos_dict.keys())))
  # get all scores in the desired format
  return [(
    numpy.array(neg_dict[probe_name], numpy.float64) if probe_name in neg_dict else None,
    numpy.array(pos_dict[probe_name], numpy.float64) if probe_name in pos_dict else None
  ) for probe_name in probe_names]
