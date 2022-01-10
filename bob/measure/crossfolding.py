import numpy
import scipy
from scipy import stats
import random 

def average_beta_posterior(k, l, lambda_, nb_samples):
    """Simulates the average beta posterior of a system with the provided markings

    This implementation is based on [GOUTTE-2005]_, equation 7.

    Figures of merit that are supported by this procedure are those which have
    the form :math:`v = k / (k + l)`:

    * Precision or Positive-Predictive Value (PPV): :math:`p = TP/(TP+FP)`, so
      :math:`k=TP`, :math:`l=FP`
    * Recall, Sensitivity, or True Positive Rate: :math:`r = TP/(TP+FN)`, so
      :math:`k=TP`, :math:`l=FN`
    * Specificity or True Negative Rate: :math:`s = TN/(TN+FP)`, so :math:`k=TN`,
      :math:`l=FP`
    * Accuracy: :math:`acc = TP+TN/(TP+TN+FP+FN)`, so :math:`k=TP+TN`,
      :math:`l=FP+FN`
    * Jaccard Index: :math:`j = TP/(TP+FP+FN)`, so :math:`k=TP`, :math:`l=FP+FN`

    Parameters
    ----------

    k : 1D int vector
        Depends on the figure of merit being considered (see above)

    l : 1D int vector
        Depends on the figure of merit being considered (see above)

    lambda_ : float
        The parameterisation of the Beta prior to consider. Use
        :math:`\lambda=1` for a flat prior.  Use :math:`\lambda=0.5` for
        Jeffrey's prior.

    nb_samples : int
        number of generated gamma distribution values


    Returns
    -------

    variates : numpy.ndarray
        An array with size ``nb_samples`` containing a realization of equation 7.

    """
    variates = numpy.zeros(nb_samples)
    for i in range(k.size) :
        variates += numpy.random.beta(a=(k[i] + lambda_), b=(l[i] + lambda_), size=nb_samples)
    return variates / k.size

def average_beta(k, l, lambda_, coverage, nb_samples):
    scores = average_beta_posterior(k, l, lambda_, nb_samples)

    left_half = (1 - coverage) / 2  # size of excluded (half) area
    sorted_scores = numpy.sort(scores)

    # n.b.: we return the equally tailed range

    # calculates position of score which would exclude the left_half (left)
    lower_index = int(round(nb_samples * left_half))

    # calculates position of score which would exclude the right_half (right)
    upper_index = int(round(nb_samples * (1 - left_half)))

    lower = sorted_scores[lower_index - 1]
    upper = sorted_scores[upper_index - 1]

    return numpy.mean(scores), scipy.stats.mode(scores)[0][0], lower, upper

def average_f1_posterior(tp, fp, fn, lambda_, nb_samples):
    variates = numpy.zeros(nb_samples)
    for i in range(tp.size) :
        u = numpy.random.gamma(shape=(tp[i] + lambda_), scale=2.0, size=nb_samples)
        v = numpy.random.gamma(
            shape=(fp[i] + fn[i] + (2 * lambda_)), scale=1.0, size=nb_samples)
        variates += u / (u + v)
    return variates / tp.size


def average_f1_score(tp, fp, fn, lambda_, coverage, nb_samples):

    scores = average_f1_posterior(tp, fp, fn, lambda_, nb_samples)

    left_half = (1 - coverage) / 2  # size of excluded (half) area
    sorted_scores = numpy.sort(scores)

    # n.b.: we return the equally tailed range

    # calculates position of score which would exclude the left_half (left)
    lower_index = int(round(nb_samples * left_half))

    # calculates position of score which would exclude the left_half (right)
    upper_index = int(round(nb_samples * (1 - left_half)))

    lower = sorted_scores[lower_index - 1]
    upper = sorted_scores[upper_index - 1]

    return numpy.mean(scores), scipy.stats.mode(scores)[0][0], lower, upper


def split_number(number, num_splits) : 
    # create an array with the same number num_splits times 
    result = numpy.repeat(int(number/num_splits), num_splits)
    # add the remaining to the last number
    result[result.size - 1] += number - (num_splits * result[0])
    # randomly add a value and remove it
    for i in range(int(result.size / 2)) : 
        rand = random.randint(0, result[2*i])
        result[2*i] = result[2*i] - rand
        result[2*i + 1] = result[2*i + 1] + rand
    return result

def average_measures(tp, fp, tn, fn, lambda_, coverage, nb_samples, num_cross_folding):
    tpcross = split_number(tp, num_cross_folding)
    fpcross = split_number(fp, num_cross_folding)
    tncross = split_number(tn, num_cross_folding)
    fncross = split_number(fn, num_cross_folding)
    return (
        average_beta(tpcross, fpcross, lambda_, coverage, nb_samples), # precision
        average_beta(tpcross, fncross, lambda_, coverage, nb_samples), # recall
        average_beta(tncross, fpcross, lambda_, coverage, nb_samples), # specificity
        average_beta(tpcross + tncross, fpcross + fncross, lambda_, coverage, nb_samples), # accuracy
        average_beta(tpcross, fpcross + fncross, lambda_, coverage, nb_samples), # jaccard index
        average_f1_score(tpcross, fpcross, fncross, lambda_, coverage, nb_samples),  # f1-score
    )




print(average_measures(20, 25, 25, 30, 0.5, 0.95, 20, 4))

