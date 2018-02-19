/**
 * @date Wed Apr 20 08:02:30 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Implements the error evaluation routines
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#include <algorithm>
#include <boost/format.hpp>
#include <limits>
#include <stdexcept>

#include <bob.core/array_copy.h>
#include <bob.core/array_sort.h>
#include <bob.core/assert.h>
#include <bob.core/cast.h>
#include <bob.core/logging.h>

#include <bob.math/linsolve.h>
#include <bob.math/pavx.h>

#include "error.h"

template <typename T>
static void sort(const blitz::Array<T, 1> &a, blitz::Array<T, 1> &b,
                 bool is_sorted) {
  if (is_sorted) {
    b.reference(a);
  } else {
    bob::core::array::ccopy(a, b);
    bob::core::array::sort<T>(b);
  }
}

std::pair<double, double>
bob::measure::farfrr(const blitz::Array<double, 1> &negatives,
                     const blitz::Array<double, 1> &positives,
                     double threshold) {
  if (std::isnan(threshold)){
    bob::core::error << "Cannot compute FAR or FRR with threshold NaN.\n";
    return std::make_pair(1.,1.);
  }
  if (!negatives.size())
    throw std::runtime_error("Cannot compute FAR when no negatives are given");
  if (!positives.size())
    throw std::runtime_error("Cannot compute FRR when no positives are given");
  blitz::sizeType total_negatives = negatives.extent(blitz::firstDim);
  blitz::sizeType total_positives = positives.extent(blitz::firstDim);
  blitz::sizeType false_accepts = blitz::count(negatives >= threshold);
  blitz::sizeType false_rejects = blitz::count(positives < threshold);
  return std::make_pair(false_accepts / (double)total_negatives,
                        false_rejects / (double)total_positives);
}

std::pair<double, double>
bob::measure::precision_recall(const blitz::Array<double, 1> &negatives,
                               const blitz::Array<double, 1> &positives,
                               double threshold) {
  if (!negatives.size() || !positives.size())
    throw std::runtime_error("Cannot compute precision or recall when no "
                             "positives or no negatives are given");
  blitz::sizeType total_positives = positives.extent(blitz::firstDim);
  blitz::sizeType false_positives = blitz::count(negatives >= threshold);
  blitz::sizeType true_positives = blitz::count(positives >= threshold);
  blitz::sizeType total_classified_positives = true_positives + false_positives;
  if (!total_classified_positives)
    total_classified_positives = 1; // avoids division by zero
  if (!total_positives)
    total_positives = 1; // avoids division by zero
  return std::make_pair(true_positives / (double)(total_classified_positives),
                        true_positives / (double)(total_positives));
}

double bob::measure::f_score(const blitz::Array<double, 1> &negatives,
                             const blitz::Array<double, 1> &positives,
                             double threshold, double weight) {
  std::pair<double, double> ratios =
      bob::measure::precision_recall(negatives, positives, threshold);
  double precision = ratios.first;
  double recall = ratios.second;
  if (weight <= 0)
    weight = 1;
  if (precision == 0 && recall == 0)
    return 0;
  return (1 + weight * weight) * precision * recall /
         (weight * weight * precision + recall);
}

double eer_predicate(double far, double frr) { return std::abs(far - frr); }

double bob::measure::eerThreshold(const blitz::Array<double, 1> &negatives,
                                  const blitz::Array<double, 1> &positives,
                                  bool is_sorted) {
  blitz::Array<double, 1> neg, pos;
  sort(negatives, neg, is_sorted);
  sort(positives, pos, is_sorted);
  return bob::measure::minimizingThreshold(neg, pos, eer_predicate);
}

double bob::measure::eerRocch(const blitz::Array<double, 1> &negatives,
                              const blitz::Array<double, 1> &positives) {
  return bob::measure::rocch2eer(bob::measure::rocch(negatives, positives));
}

double bob::measure::farThreshold(const blitz::Array<double, 1> &negatives,
                                  const blitz::Array<double, 1> &positives,
                                  double far_value, bool is_sorted) {
  // check the parameters are valid
  if (far_value < 0. || far_value > 1.) {
    boost::format m("the argument for `far_value' cannot take the value %f - "
                    "the value must be in the interval [0.,1.]");
    m % far_value;
    throw std::runtime_error(m.str());
  }
  if (negatives.size() < 2) {
    throw std::runtime_error(
        "the number of negative scores must be at least 2");
  }

  // sort the negatives array, if necessary, and keep it in the scores variable
  blitz::Array<double, 1> scores;
  sort(negatives, scores, is_sorted);

  double epsilon = std::numeric_limits<double>::epsilon();
  // handle special case of far == 1 without any iterating
  if (far_value >= 1 - epsilon)
    return std::nextafter(scores(0), scores(0)-1);

  // Reverse negatives so the end is the start. This way the code below will be
  // very similar to the implementation in the frrThreshold function. The
  // implementations are not exactly the same though.
  scores.reverseSelf(0);
  // Move towards the end of array changing the threshold until we pass the
  // desired FAR value. Starting with a threshold that corresponds to FAR == 0.
  int total_count = scores.extent(0);
  int current_position = 0;
  // since the comparison is `if score >= threshold then accept as genuine`, we
  // can choose the largest score value + eps as the threshold so that we can
  // get for 0% FAR.
  double valid_threshold = std::nextafter(scores(current_position), scores(current_position)+1);
  double current_threshold;
  double future_far;
  while (current_position < total_count) {
    current_threshold = scores(current_position);
    // keep iterating if values are repeated
    while (current_position < total_count-1 && scores(current_position+1) == current_threshold)
      current_position++;
    // All the scores up to the current position and including the current
    // position will be accepted falsely.
    future_far = (double)(current_position+1) / (double)total_count;
    if (future_far > far_value)
      break;
    valid_threshold = current_threshold;
    current_position++;
  }
  return valid_threshold;
}

double bob::measure::frrThreshold(const blitz::Array<double, 1> &negatives,
                                  const blitz::Array<double, 1> &positives,
                                  double frr_value, bool is_sorted) {

  // check the parameters are valid
  if (frr_value < 0. || frr_value > 1.) {
    boost::format m("the argument for `frr_value' cannot take the value %f - "
                    "the value must be in the interval [0.,1.]");
    m % frr_value;
    throw std::runtime_error(m.str());
  }
  if (positives.size() < 2) {
    throw std::runtime_error(
        "the number of positive scores must be at least 2");
  }

  // sort the positives array, if necessary, and keep it in the scores variable
  blitz::Array<double, 1> scores;
  sort(positives, scores, is_sorted);

  double epsilon = std::numeric_limits<double>::epsilon();
  // handle special case of frr == 1 without any iterating
  if (frr_value >= 1 - epsilon)
    return std::nextafter(scores(scores.extent(0)-1), scores(scores.extent(0)-1)+1);

  // Move towards the end of array changing the threshold until we pass the
  // desired FRR value. Starting with a threshold that corresponds to FRR == 0.
  int total_count = scores.extent(0);
  int current_position = 0;
  // since the comparison is `if score >= threshold then accept as genuine`, we
  // can use the smallest positive score as the threshold for 0% FRR.
  double valid_threshold = scores(current_position);
  double current_threshold;
  double future_frr;
  while (current_position < total_count) {
    current_threshold = scores(current_position);
    // keep iterating if values are repeated
    while (current_position < total_count-1 && scores(current_position+1) == current_threshold)
      current_position++;
    // All the scores up to the current_position but not including
    // current_position will be rejected falsely.
    future_frr = (double)current_position / (double)total_count;
    if (future_frr > frr_value)
      break;
    valid_threshold = current_threshold;
    current_position++;
  }
  return valid_threshold;
}

/**
 * Provides a functor predicate for weighted error calculation
 */
class weighted_error {

  double m_weight; ///< The weighting factor

public: // api
  weighted_error(double weight) : m_weight(weight) {
    if (weight > 1.0)
      m_weight = 1.0;
    if (weight < 0.0)
      m_weight = 0.0;
  }

  inline double operator()(double far, double frr) const {
    return (m_weight * far) + ((1.0 - m_weight) * frr);
  }
};

double bob::measure::minWeightedErrorRateThreshold(
    const blitz::Array<double, 1> &negatives,
    const blitz::Array<double, 1> &positives, double cost, bool is_sorted) {
  blitz::Array<double, 1> neg, pos;
  sort(negatives, neg, is_sorted);
  sort(positives, pos, is_sorted);
  weighted_error predicate(cost);
  return bob::measure::minimizingThreshold(neg, pos, predicate);
}

blitz::Array<double, 2>
bob::measure::roc(const blitz::Array<double, 1> &negatives,
                  const blitz::Array<double, 1> &positives, size_t points) {
  // Uses roc_for_far internally
  // Create an far_list
  blitz::Array<double, 1> far_list((int)points);
  int min_far = -8;  // minimum FAR in terms of 10^(min_far)
  double counts_per_step = points / (-min_far) ;
  for (int i = 1-(int)points; i <= 0; ++i) {
    far_list(i+(int)points-1) = std::pow(10., (double)i/counts_per_step);
  }
  return bob::measure::roc_for_far(negatives, positives, far_list, false);
}

blitz::Array<double, 2>
bob::measure::precision_recall_curve(const blitz::Array<double, 1> &negatives,
                                     const blitz::Array<double, 1> &positives,
                                     size_t points) {
  double min = std::min(blitz::min(negatives), blitz::min(positives));
  double max = std::max(blitz::max(negatives), blitz::max(positives));
  double step = (max - min) / ((double)points - 1.0);
  blitz::Array<double, 2> retval(2, points);
  for (int i = 0; i < (int)points; ++i) {
    std::pair<double, double> ratios =
        bob::measure::precision_recall(negatives, positives, min + i * step);
    retval(0, i) = ratios.first;
    retval(1, i) = ratios.second;
  }
  return retval;
}

/**
  * Structure for getting permutations when sorting an array
  */
struct ComparePairs {
  ComparePairs(const blitz::Array<double, 1> &v) : m_v(v) {}

  bool operator()(size_t a, size_t b) { return m_v(a) < m_v(b); }

  blitz::Array<double, 1> m_v;
};

/**
  * Sort an array and get the permutations (using stable_sort)
  */
void sortWithPermutation(const blitz::Array<double, 1> &values,
                         std::vector<size_t> &v) {
  int N = values.extent(0);
  bob::core::array::assertSameDimensionLength(N, v.size());
  for (int i = 0; i < N; ++i)
    v[i] = i;

  std::stable_sort(v.begin(), v.end(), ComparePairs(values));
}

blitz::Array<double, 2>
bob::measure::rocch(const blitz::Array<double, 1> &negatives,
                    const blitz::Array<double, 1> &positives) {
  // Number of positive and negative scores
  size_t Nt = positives.extent(0);
  size_t Nn = negatives.extent(0);
  size_t N = Nt + Nn;

  // Create a big array with all scores
  blitz::Array<double, 1> scores(N);
  blitz::Range rall = blitz::Range::all();
  scores(blitz::Range(0, Nt - 1)) = positives(rall);
  scores(blitz::Range(Nt, N - 1)) = negatives(rall);

  // It is important here that scores that are the same (i.e. already in order)
  // should NOT be swapped.
  // std::stable_sort has this property.
  std::vector<size_t> perturb(N);
  sortWithPermutation(scores, perturb);

  // Apply permutation
  blitz::Array<size_t, 1> Pideal(N);
  for (size_t i = 0; i < N; ++i)
    Pideal(i) = (perturb[i] < Nt ? 1 : 0);
  blitz::Array<double, 1> Pideal_d = bob::core::array::cast<double>(Pideal);

  // Apply the PAVA algorithm
  blitz::Array<double, 1> Popt(N);
  blitz::Array<size_t, 1> width = bob::math::pavxWidth(Pideal_d, Popt);

  // Allocate output
  int nbins = width.extent(0);
  blitz::Array<double, 2> retval(2, nbins + 1); // FAR, FRR

  // Fill in output
  size_t left = 0;
  size_t fa = Nn;
  size_t miss = 0;
  for (int i = 0; i < nbins; ++i) {
    retval(0, i) = fa / (double)Nn;   // pfa
    retval(1, i) = miss / (double)Nt; // pmiss
    left += width(i);
    if (left >= 1)
      miss = blitz::sum(Pideal(blitz::Range(0, left - 1)));
    else
      miss = 0;
    if (Pideal.extent(0) - 1 >= (int)left)
      fa = N - left -
           blitz::sum(Pideal(blitz::Range(left, Pideal.extent(0) - 1)));
    else
      fa = 0;
  }
  retval(0, nbins) = fa / (double)Nn;   // pfa
  retval(1, nbins) = miss / (double)Nt; // pmiss

  return retval;
}

double bob::measure::rocch2eer(const blitz::Array<double, 2> &pfa_pmiss) {
  bob::core::array::assertSameDimensionLength(2, pfa_pmiss.extent(0));
  const int N = pfa_pmiss.extent(1);

  double eer = 0.;
  blitz::Array<double, 2> XY(2, 2);
  blitz::Array<double, 1> one(2);
  one = 1.;
  blitz::Array<double, 1> seg(2);
  double &XY00 = XY(0, 0);
  double &XY01 = XY(0, 1);
  double &XY10 = XY(1, 0);
  double &XY11 = XY(1, 1);

  double eerseg = 0.;
  for (int i = 0; i < N - 1; ++i) {
    // Define XY matrix
    XY00 = pfa_pmiss(0, i);     // pfa
    XY10 = pfa_pmiss(0, i + 1); // pfa
    XY01 = pfa_pmiss(1, i);     // pmiss
    XY11 = pfa_pmiss(1, i + 1); // pmiss
    // xx and yy should be sorted:
    assert(XY10 <= XY00 && XY01 <= XY11);

    // Commpute "dd"
    double abs_dd0 = std::fabs(XY00 - XY10);
    double abs_dd1 = std::fabs(XY01 - XY11);
    if (std::min(abs_dd0, abs_dd1) < std::numeric_limits<double>::epsilon())
      eerseg = 0.;
    else {
      // Find line coefficients seg s.t. XY.seg = 1,
      bob::math::linsolve_(XY, one, seg);
      // Candidate for the EER (to be compared to current value)
      eerseg = 1. / blitz::sum(seg);
    }

    eer = std::max(eer, eerseg);
  }

  return eer;
}

/**
 * This function computes the ROC coordinates for the given positive and
 * negative values at the given FAR positions.
 *
 * @param negatives Impostor scores
 * @param positives Client scores
 * @param far_list  The list of FAR values where the FRR should be calculated
 *
 * @return The ROC curve with the FAR in the first row and the FRR in the
 * second.
 */
blitz::Array<double, 2>
bob::measure::roc_for_far(const blitz::Array<double, 1> &negatives,
                          const blitz::Array<double, 1> &positives,
                          const blitz::Array<double, 1> &far_list,
                          bool is_sorted) {
  int n_points = far_list.extent(0);

  if (negatives.extent(0) == 0)
    throw std::runtime_error("The given set of negatives is empty.");
  if (positives.extent(0) == 0)
    throw std::runtime_error("The given set of positives is empty.");

  // sort negative and positive scores ascendantly
  blitz::Array<double, 1> neg, pos;
  sort(negatives, neg, is_sorted);
  sort(positives, pos, is_sorted);

  blitz::Array<double, 2> retval(2, n_points);

  // index into the FAR list
  int far_index = n_points - 1;

  // Get the threshold for the requested far values and calculate far and frr
  // values based on the threshold.
  while(far_index >= 0) {
    // calculate the threshold for the requested far
    auto threshold = bob::measure::farThreshold(neg, pos, far_list(far_index), true);
    // calculate the frr and re-calculate the far
    auto farfrr = bob::measure::farfrr(neg, pos, threshold);
    retval(0, far_index) = farfrr.first;
    retval(1, far_index) = farfrr.second;
    far_index--;
  }

  return retval;
}

/**
 * The input to this function is a cumulative probability.  The output from
 * this function is the Normal deviate that corresponds to that probability.
 * For example:
 *
 *  INPUT | OUTPUT
 * -------+--------
 *  0.001 | -3.090
 *  0.01  | -2.326
 *  0.1   | -1.282
 *  0.5   |  0.0
 *  0.9   |  1.282
 *  0.99  |  2.326
 *  0.999 |  3.090
 */
static double _ppndf(double p) {
  // some constants we need for the calculation.
  // these come from the NIST implementation...
  static const double SPLIT = 0.42;
  static const double A0 = 2.5066282388;
  static const double A1 = -18.6150006252;
  static const double A2 = 41.3911977353;
  static const double A3 = -25.4410604963;
  static const double B1 = -8.4735109309;
  static const double B2 = 23.0833674374;
  static const double B3 = -21.0622410182;
  static const double B4 = 3.1308290983;
  static const double C0 = -2.7871893113;
  static const double C1 = -2.2979647913;
  static const double C2 = 4.8501412713;
  static const double C3 = 2.3212127685;
  static const double D1 = 3.5438892476;
  static const double D2 = 1.6370678189;
  static const double eps = 2.2204e-16;

  double retval;

  if (p >= 1.0)
    p = 1 - eps;
  if (p <= 0.0)
    p = eps;

  double q = p - 0.5;

  if (std::abs(q) <= SPLIT) {
    double r = q * q;
    retval = q * (((A3 * r + A2) * r + A1) * r + A0) /
             ((((B4 * r + B3) * r + B2) * r + B1) * r + 1.0);
  } else {
    // r = sqrt (log (0.5 - abs(q)));
    double r = (q > 0.0 ? 1.0 - p : p);
    if (r <= 0.0)
      throw std::runtime_error("measure::ppndf(): r <= 0.0!");
    r = sqrt((-1.0) * log(r));
    retval = (((C3 * r + C2) * r + C1) * r + C0) / ((D2 * r + D1) * r + 1.0);
    if (q < 0)
      retval *= -1.0;
  }

  return retval;
}

namespace blitz {
BZ_DECLARE_FUNCTION(_ppndf) ///< A blitz::Array binding
}

double bob::measure::ppndf(double value) { return _ppndf(value); }

blitz::Array<double, 2>
bob::measure::det(const blitz::Array<double, 1> &negatives,
                  const blitz::Array<double, 1> &positives, size_t points) {
  blitz::Array<double, 2> retval(2, points);
  retval = blitz::_ppndf(bob::measure::roc(negatives, positives, points));
  return retval;
}

blitz::Array<double, 2>
bob::measure::epc(const blitz::Array<double, 1> &dev_negatives,
                  const blitz::Array<double, 1> &dev_positives,
                  const blitz::Array<double, 1> &test_negatives,
                  const blitz::Array<double, 1> &test_positives, size_t points,
                  bool is_sorted, bool thresholds) {

  blitz::Array<double, 1> dev_neg, dev_pos;
  sort(dev_negatives, dev_neg, is_sorted);
  sort(dev_positives, dev_pos, is_sorted);

  double step = 1.0 / ((double)points - 1.0);
  auto retval_shape0 = (thresholds) ? 3 : 2;
  blitz::Array<double, 2> retval(retval_shape0, points);
  for (int i = 0; i < (int)points; ++i) {
    double alpha = (double)i * step;
    retval(0, i) = alpha;
    double threshold = bob::measure::minWeightedErrorRateThreshold(
        dev_neg, dev_pos, alpha, true);
    std::pair<double, double> ratios =
        bob::measure::farfrr(test_negatives, test_positives, threshold);
    retval(1, i) = (ratios.first + ratios.second) / 2;
    if (thresholds) {
      retval(2, i) = threshold;
    }
  }
  return retval;
}
