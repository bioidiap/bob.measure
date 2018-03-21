import numpy

def remove_nan(scores):
    """removes the NaNs from the scores"""
    nans = numpy.isnan(scores)
    sum_nans = sum(nans)
    total = len(scores)
    if sum_nans > 0:
        logger.warning('Found {} NaNs in {} scores'.format(sum_nans, total))
    return scores[numpy.where(~nans)], sum_nans, total


def get_fta(scores):
  """calculates the Failure To Acquire (FtA) rate"""
  fta_sum, fta_total = 0, 0
  neg, sum_nans, total = remove_nan(scores[0])
  fta_sum += sum_nans
  fta_total += total
  pos, sum_nans, total = remove_nan(scores[1])
  fta_sum += sum_nans
  fta_total += total
  return ((neg, pos), fta_sum / fta_total)


