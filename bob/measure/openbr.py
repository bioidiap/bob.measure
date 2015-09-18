"""This file includes functionality to convert between Bob's four column or five column score files and the Matrix files used in OpenBR."""

import numpy
import sys
import logging
logger = logging.getLogger("bob.measure")

from .load import open_file, four_column, five_column

def write_matrix(
    score_file,
    matrix_file,
    mask_file,
    model_names = None,
    probe_names = None,
    score_file_format = '4column',
    gallery_file_name = 'unknown-gallery.lst',
    probe_file_name = 'unknown-probe.lst',
    search = None
):
  """Writes the OpenBR matrix and mask files (version 2), given the score file.
  If gallery and probe names are provided, the matrices in both files will be sorted by gallery and probe names.
  Otherwise, the order will be the same as given in the score file.

  If ``search`` is given (as an integer), the resulting matrix files will be in the *search* format, keeping the given number of gallery scores with the highest values for each probe.

  .. warning::
    When provided with a 4-column score file, this function will work only, if there is only a single model id for each client.

  Keyword parameters:

  score_file : str
    The 4 or 5 column style score file written by bob.

  matrix_file : str
    The OpenBR matrix file that should be written.
    Usually, the file name extension is ``.mtx``

  mask_file : str
    The OpenBR mask file that should be written.
    The mask file defines, which values are positives, negatives or to be ignored.
    Usually, the file name extension is ``.mask``

  model_names : [str] or ``None``
    If given, the matrix will be written in the same order as the given model names.
    The model names must be identical with the second column in the 5-column ``score_file``.

    .. note::
       If the score file is in four column format, the model_names must be the client ids stored in the first column.
       In this case, there might be only a single model per client

    Only the scores of the given models will be considered.

  probe_names : [str] or ``None``
    If given, the matrix will be written in the same order as the given probe names (the ``path`` of the probe).
    The probe names are identical to the third column of the 4-column (or the fourth column of the 5-column) ``score_file``.
    Only the scores of the given probe names will be considered in this case.

  score_file_format : one of ``('4column', '5column')``
    The format, in which the ``score_file`` is.

  gallery_file_name : str
    The name of the gallery file that will be written in the header of the OpenBR files.

  probe_file_name : str
    The name of the probe file that will be written in the header of the OpenBR files.

  search : int or ``None``
    If given, the scores will be sorted per probe, keeping the specified number of highest scores.
    If the given number is higher than the models, ``NaN`` values will be added, and the mask will contain ``0x00`` values.
  """

  def _write_matrix(filename, matrix):
    ## Helper function to write a matrix file as required by OpenBR
    with open(filename, 'wb') as f:
      # write the first four lines
      header = "S2\n%s\n%s\nM%s %d %d " % (gallery_file_name, probe_file_name, 'B' if matrix.dtype == numpy.uint8 else 'F', matrix.shape[0], matrix.shape[1])
      footer = "\n"
      if sys.version_info[0] > 2: header, footer = header.encode('utf-8'), footer.encode('utf-8')
      f.write(header)
      # write magic number
      numpy.array(0x12345678, numpy.int32).tofile(f)
      f.write(footer)
      # write the matrix
      matrix.tofile(f)


  # define read functions, and which information should be read
  read_function = {'4column' : four_column, '5column' : five_column}[score_file_format]
  offset = {'4column' : 0, '5column' : 1}[score_file_format]

  # first, read the score file and estimate model and probe names, if not given
  if model_names is None or probe_names is None:
    model_names, probe_names = [], []
    model_set, probe_set = set(), set()

    # read the score file
    for line in read_function(score_file):
      model, probe = line[offset], line[2+offset]
      if model not in model_set:
        model_names.append(model)
        model_set.add(model)
      if probe not in probe_set:
        probe_names.append(probe)
        probe_set.add(probe)

  if search is None:
    # create a shortcut to get indices for client and probe subset (to increase speed)
    model_dict, probe_dict = {}, {}
    for i,m in enumerate(model_names): model_dict[m]=i
    for i,p in enumerate(probe_names): probe_dict[p]=i

    # create the matrices in the desired size
    matrix = numpy.ndarray((len(probe_names), len(model_names)), numpy.float32)
    matrix[:] = numpy.nan
    mask = numpy.zeros(matrix.shape, numpy.uint8)

    # now, iterate through the score file and fill in the matrix
    for line in read_function(score_file):
      client, model, id, probe, score = line[0], line[offset], line[1+offset], line[2+offset], line[3+offset]

      assert model in model_dict
      assert probe in probe_dict

      model_index = model_dict[model]
      probe_index = probe_dict[probe]

      # check, if we have already written something into that matrix element
      if mask[probe_index, model_index]:
        logger.warn("Overwriting existing matrix '%f' element of client '%s' and probe '%s' with '%f'", matrix[probe_index, model_index], client, probe, score)

      matrix[probe_index, model_index] = score
      mask[probe_index, model_index] = 0xff if client == id else 0x7f

  else:
    # get the correct search parameter, if negative
    if search < 0:
      search = len(model_names)

    # create the matrices in the desired size
    matrix = numpy.ndarray((len(probe_names), search), numpy.float32)
    matrix[:] = numpy.nan
    mask = numpy.zeros(matrix.shape, numpy.uint8)

    # get the scores, sorted by probe
    scores = {}
    for line in read_function(score_file):
      client, model, id, probe, score = line[0], line[offset], line[1+offset], line[2+offset], line[3+offset]

      if probe not in scores:
        scores[probe] = []
      scores[probe].append((score, 0xff if client == id else 0x7f))

    # go ahead and sort the scores per probe
    sorted_scores = {}
    for k,v in scores.items(): sorted_scores[k] = sorted(v, key=lambda x: x[0], reverse=True)

    # now, write matrix
    for p, probe in enumerate(probe_names):
      if probe in scores:
        for m in range(min(search, len(sorted_scores[probe]))):
          matrix[p,m], mask[p,m] = sorted_scores[probe][m]


  # OK, now finally write the file in the desired format
  _write_matrix(mask_file, mask)
  _write_matrix(matrix_file, matrix)
