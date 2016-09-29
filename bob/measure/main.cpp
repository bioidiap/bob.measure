/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri 25 Oct 16:54:55 2013
 *
 * @brief Bindings to bob::measure
 */

#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif
#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob.core/api.h>
#include <bob.io.base/api.h>
#include <bob.extension/documentation.h>

#include "cpp/error.h"

static int double1d_converter(PyObject* o, PyBlitzArrayObject** a) {
  if (PyBlitzArray_Converter(o, a) == 0) return 0;
  // in this case, *a is set to a new reference
  if ((*a)->type_num != NPY_FLOAT64 || (*a)->ndim != 1) {
    PyErr_Format(PyExc_TypeError, "cannot convert blitz::Array<%s,%" PY_FORMAT_SIZE_T "d> to a blitz::Array<double,1>", PyBlitzArray_TypenumAsString((*a)->type_num), (*a)->ndim);
    return 0;
  }
  return 1;
}

static int double2d_converter(PyObject* o, PyBlitzArrayObject** a) {
  if (PyBlitzArray_Converter(o, a) == 0) return 0;
  // in this case, *a is set to a new reference
  if ((*a)->type_num != NPY_FLOAT64 || (*a)->ndim != 2) {
    PyErr_Format(PyExc_TypeError, "cannot convert blitz::Array<%s,%" PY_FORMAT_SIZE_T "d> to a blitz::Array<double,2>", PyBlitzArray_TypenumAsString((*a)->type_num), (*a)->ndim);
    return 0;
  }
  return 1;
}


static auto epc_doc = bob::extension::FunctionDoc(
  "epc",
  "Calculates points of an Expected Performance Curve (EPC)",
  "Calculates the EPC curve given a set of positive and negative scores and a desired number of points. "
  "Returns a two-dimensional :py:class:`numpy.ndarray` of type float that express the X (cost) and Y (weighted error rare on the test set given the min. threshold on the development set) coordinates in this order. "
  "Please note that, in order to calculate the EPC curve, one needs two sets of data comprising a development set and a test set. "
  "The minimum weighted error is calculated on the development set and then applied to the test set to evaluate the weighted error rate at that position.\n\n"
  "The EPC curve plots the HTER on the test set for various values of 'cost'. "
  "For each value of 'cost', a threshold is found that provides the minimum weighted error (see :py:func:`bob.measure.min_weighted_error_rate_threshold`) on the development set. "
  "Each threshold is consecutively applied to the test set and the resulting weighted error values are plotted in the EPC.\n\n"
  "The cost points in which the EPC curve are calculated are distributed uniformly in the range :math:`[0.0, 1.0]`.\n\n"
  ".. note:: It is more memory efficient, when sorted arrays of scores are provided and the ``is_sorted`` parameter is set to ``True``."
)
.add_prototype("dev_negatives, dev_positives, test_negatives, test_positives, n_points, is_sorted", "curve")
.add_parameter("dev_negatives, dev_positives, test_negatives, test_positives", "array_like(1D, float)", "The scores for negatives and positives of the development and test set")
.add_parameter("n_points", "int", "The number of weights for which the EPC curve should be computed")
.add_parameter("is_sorted", "bool", "[Default: ``False``] Set this to ``True`` if the scores are already sorted. If ``False``, scores will be sorted internally, which will require more memory")
.add_return("curve", "array_like(2D, float)", "The EPC curve, with the first row containing the weights, and the second row containing the weighted thresholds on the test set")
;
static PyObject* epc(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = epc_doc.kwlist();

  PyBlitzArrayObject* dev_neg;
  PyBlitzArrayObject* dev_pos;
  PyBlitzArrayObject* test_neg;
  PyBlitzArrayObject* test_pos;
  Py_ssize_t n_points;
  PyObject* is_sorted = Py_False;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&O&O&n|O",
        kwlist,
        &double1d_converter, &dev_neg,
        &double1d_converter, &dev_pos,
        &double1d_converter, &test_neg,
        &double1d_converter, &test_pos,
        &n_points,
        &is_sorted
        )) return 0;

  //protects acquired resources through this scope
  auto dev_neg_ = make_safe(dev_neg);
  auto dev_pos_ = make_safe(dev_pos);
  auto test_neg_ = make_safe(test_neg);
  auto test_pos_ = make_safe(test_pos);

  auto result = bob::measure::epc(
      *PyBlitzArrayCxx_AsBlitz<double,1>(dev_neg),
      *PyBlitzArrayCxx_AsBlitz<double,1>(dev_pos),
      *PyBlitzArrayCxx_AsBlitz<double,1>(test_neg),
      *PyBlitzArrayCxx_AsBlitz<double,1>(test_pos),
      n_points, PyObject_IsTrue(is_sorted));

  return PyBlitzArrayCxx_AsNumpy(result);
BOB_CATCH_FUNCTION("epc", 0)
}

static auto det_doc = bob::extension::FunctionDoc(
  "det",
  "Calculates points of an Detection Error-Tradeoff (DET) curve",
  "Calculates the DET curve given a set of negative and positive scores and a desired number of points. Returns a two-dimensional array of doubles that express on its rows:\n\n"
  "[0]  X axis values in the normal deviate scale for the false-accepts\n\n"
  "[1]  Y axis values in the normal deviate scale for the false-rejections\n\n"
  "You can plot the results using your preferred tool to first create a plot using rows 0 and 1 from the returned value and then replace the X/Y axis annotation using a pre-determined set of tickmarks as recommended by NIST. "
  "The derivative scales are computed with the :py:func:`bob.measure.ppndf` function."
)
.add_prototype("negatives, positives, n_points", "curve")
.add_parameter("negatives, positives", "array_like(1D, float)", "The list of negative and positive scores to compute the DET for")
.add_parameter("n_points", "int", "The number of points on the DET curve, for which the DET should be evaluated")
.add_return("curve", "array_like(2D, float)", "The DET curve, with the FAR in the first and the FRR in the second row")
;
static PyObject* det(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  char** kwlist = det_doc.kwlist();

  PyBlitzArrayObject* neg;
  PyBlitzArrayObject* pos;
  Py_ssize_t n_points;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&n",
        kwlist,
        &double1d_converter, &neg,
        &double1d_converter, &pos,
        &n_points
        )) return 0;

  //protects acquired resources through this scope
  auto neg_ = make_safe(neg);
  auto pos_ = make_safe(pos);

  auto result = bob::measure::det(
      *PyBlitzArrayCxx_AsBlitz<double,1>(neg),
      *PyBlitzArrayCxx_AsBlitz<double,1>(pos),
      n_points);

  return PyBlitzArrayCxx_AsNumpy(result);
BOB_CATCH_FUNCTION("det", 0)
}

static auto ppndf_doc = bob::extension::FunctionDoc(
  "ppndf",
  "Returns the Deviate Scale equivalent of a false rejection/acceptance ratio",
  "The algorithm that calculates the deviate scale is based on function ppndf() from the NIST package DETware version 2.1, freely available on the internet. "
  "Please consult it for more details. "
  "By 20.04.2011, you could find such package `here <http://www.itl.nist.gov/iad/mig/tools/>`_."
)
.add_prototype("value", "ppndf")
.add_parameter("value", "float", "The value (usually FAR or FRR) for which the ppndf should be calculated")
.add_return("ppndf", "float", "The derivative scale of the given value")
;
static PyObject* ppndf(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  char** kwlist = ppndf_doc.kwlist();
  double v;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "d", kwlist, &v)) return 0;

  return Py_BuildValue("d", bob::measure::ppndf(v));
BOB_CATCH_FUNCTION("ppndf", 0)
}

static auto roc_doc = bob::extension::FunctionDoc(
  "roc",
  "Calculates points of an Receiver Operating Characteristic (ROC)",
  "Calculates the ROC curve given a set of negative and positive scores and a desired number of points. "
)
.add_prototype("negatives, positives, n_points", "curve")
.add_parameter("negatives, positives", "array_like(1D, float)", "The negative and positive scores, for which the ROC curve should be calculated")
.add_parameter("n_points", "int", "The number of points, in which the ROC curve are calculated, which are distributed uniformly in the range ``[min(negatives, positives), max(negatives, positives)]``")
.add_return("curve", "array_like(2D, float)", "A two-dimensional array of doubles that express the X (FAR) and Y (FRR) coordinates in this order")
;
static PyObject* roc(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  static char** kwlist = roc_doc.kwlist();

  PyBlitzArrayObject* neg;
  PyBlitzArrayObject* pos;
  Py_ssize_t n_points;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&n",
        kwlist,
        &double1d_converter, &neg,
        &double1d_converter, &pos,
        &n_points
        )) return 0;

  //protects acquired resources through this scope
  auto neg_ = make_safe(neg);
  auto pos_ = make_safe(pos);

  auto result = bob::measure::roc(
      *PyBlitzArrayCxx_AsBlitz<double,1>(neg),
      *PyBlitzArrayCxx_AsBlitz<double,1>(pos),
      n_points);

  return PyBlitzArrayCxx_AsNumpy(result);
BOB_CATCH_FUNCTION("roc", 0)
}

static auto farfrr_doc = bob::extension::FunctionDoc(
  "farfrr",
  "Calculates the false-acceptance (FA) ratio and the false-rejection (FR) ratio for the given positive and negative scores and a score threshold",
  "``positives`` holds the score information for samples that are labeled to belong to a certain class (a.k.a., 'signal' or 'client'). "
  "``negatives`` holds the score information for samples that are labeled **not** to belong to the class (a.k.a., 'noise' or 'impostor'). "
  "It is expected that 'positive' scores are, at least by design, greater than 'negative' scores. "
  "So, every 'positive' value that falls bellow the threshold is considered a false-rejection (FR). "
  "`negative` samples that fall above the threshold are considered a false-accept (FA).\n\n"
  "Positives that fall on the threshold (exactly) are considered correctly classified. "
  "Negatives that fall on the threshold (exactly) are considered **incorrectly** classified. "
  "This equivalent to setting the comparison like this pseudo-code:\n\n"
  "  ``foreach (positive as K) if K < threshold: falseRejectionCount += 1``\n\n"
  "  ``foreach (negative as K) if K >= threshold: falseAcceptCount += 1``\n\n"
  "The output is in form of a tuple of two double-precision real numbers. "
  "The numbers range from 0 to 1. "
  "The first element of the pair is the false-accept ratio (FAR), the second element the false-rejection ratio (FRR).\n\n"
  "The ``threshold`` value does not necessarily have to fall in the range covered by the input scores (negatives and positives altogether), but if it does not, the output will be either (1.0, 0.0) or (0.0, 1.0), depending on the side the threshold falls.\n\n"
  "It is possible that scores are inverted in the negative/positive sense. "
  "In some setups the designer may have setup the system so 'positive' samples have a smaller score than the 'negative' ones. "
  "In this case, make sure you normalize the scores so positive samples have greater scores before feeding them into this method."
)
.add_prototype("negatives, positives, threshold", "far, frr")
.add_parameter("negatives", "array_like(1D, float)", "The scores for comparisons of objects of different classes")
.add_parameter("positives", "array_like(1D, float)", "The scores for comparisons of objects of the same class")
.add_parameter("threshold", "float", "The threshold to separate correctly and incorrectly classified scores")
.add_return("far", "float", "The False Accept Rate (FAR) for the given threshold")
.add_return("frr", "float", "The False Reject Rate (FRR) for the given threshold")
;
static PyObject* farfrr(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  char** kwlist = farfrr_doc.kwlist();

  PyBlitzArrayObject* neg;
  PyBlitzArrayObject* pos;
  double threshold;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&d",
        kwlist,
        &double1d_converter, &neg,
        &double1d_converter, &pos,
        &threshold
        )) return 0;

  //protects acquired resources through this scope
  auto neg_ = make_safe(neg);
  auto pos_ = make_safe(pos);

  auto result = bob::measure::farfrr(
      *PyBlitzArrayCxx_AsBlitz<double,1>(neg),
      *PyBlitzArrayCxx_AsBlitz<double,1>(pos),
      threshold);

  return Py_BuildValue("dd", result.first, result.second);
BOB_CATCH_FUNCTION("farfrr", 0)
}

static auto eer_threshold_doc = bob::extension::FunctionDoc(
  "eer_threshold",
  "Calculates the threshold that is as close as possible to the equal-error-rate (EER) for the given input data",
  "The EER should be the point where the FAR equals the FRR. "
  "Graphically, this would be equivalent to the intersection between the ROC (or DET) curves and the identity.\n\n"
  ".. note::\n\n"
  "   The scores will be sorted internally, requiring the scores to be copied.\n"
  "   To avoid this copy, you can sort both sets of scores externally in ascendant order, and set the ``is_sorted`` parameter to ``True``"
)
.add_prototype("negatives, positives, [is_sorted]", "threshold")
.add_parameter("negatives, positives", "array_like(1D, float)", "The set of negative and positive scores to compute the threshold")
.add_parameter("is_sorted", "bool", "[Default: ``False``] Are both sets of scores already in ascendantly sorted order?")
.add_return("threshold", "float", "The threshold (i.e., as used in :py:func:`bob.measure.farfrr`) where FAR and FRR are as close as possible")
;
static PyObject* eer_threshold(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  char** kwlist = eer_threshold_doc.kwlist();

  PyBlitzArrayObject* neg;
  PyBlitzArrayObject* pos;
  PyObject* is_sorted = Py_False;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&|O",
        kwlist,
        &double1d_converter, &neg,
        &double1d_converter, &pos,
        &is_sorted
        )) return 0;

  //protects acquired resources through this scope
  auto neg_ = make_safe(neg);
  auto pos_ = make_safe(pos);

  double result = bob::measure::eerThreshold(
      *PyBlitzArrayCxx_AsBlitz<double,1>(neg),
      *PyBlitzArrayCxx_AsBlitz<double,1>(pos),
      PyObject_IsTrue(is_sorted));

  return Py_BuildValue("d", result);
BOB_CATCH_FUNCTION("eer_threshold", 0)
}

static auto min_weighted_error_rate_threshold_doc = bob::extension::FunctionDoc(
  "min_weighted_error_rate_threshold",
  "Calculates the threshold that minimizes the error rate for the given input data",
  "The ``cost`` parameter determines the relative importance between false-accepts and false-rejections. "
  "This number should be between 0 and 1 and will be clipped to those extremes. "
  "The value to minimize becomes: :math:`ER_{cost} = cost * FAR + (1-cost) * FRR`. "
  "The higher the cost, the higher the importance given to **not** making mistakes classifying negatives/noise/impostors.\n\n"
  ".. note:: "
  "The scores will be sorted internally, requiring the scores to be copied. "
  "To avoid this copy, you can sort both sets of scores externally in ascendant order, and set the ``is_sorted`` parameter to ``True``"
)
.add_prototype("negatives, positives, cost, [is_sorted]", "threshold")
.add_parameter("negatives, positives", "array_like(1D, float)", "The set of negative and positive scores to compute the threshold")
.add_parameter("cost", "float", "The relative cost over FAR with respect to FRR in the threshold calculation")
.add_parameter("is_sorted", "bool", "[Default: ``False``] Are both sets of scores already in ascendantly sorted order?")
.add_return("threshold", "float", "The threshold for which the weighted error rate is minimal")
;
static PyObject* min_weighted_error_rate_threshold(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  char** kwlist = min_weighted_error_rate_threshold_doc.kwlist();

  PyBlitzArrayObject* neg;
  PyBlitzArrayObject* pos;
  double cost;
  PyObject* is_sorted = Py_False;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&d|O",
        kwlist,
        &double1d_converter, &neg,
        &double1d_converter, &pos,
        &cost,
        &is_sorted
        )) return 0;

  //protects acquired resources through this scope
  auto neg_ = make_safe(neg);
  auto pos_ = make_safe(pos);

  double result = bob::measure::minWeightedErrorRateThreshold(
      *PyBlitzArrayCxx_AsBlitz<double,1>(neg),
      *PyBlitzArrayCxx_AsBlitz<double,1>(pos),
      cost,
      PyObject_IsTrue(is_sorted));

  return Py_BuildValue("d", result);
BOB_CATCH_FUNCTION("min_weighted_error_rate_threshold", 0)
}

static auto min_hter_threshold_doc = bob::extension::FunctionDoc(
  "min_hter_threshold",
  "Calculates the :py:func:`bob.measure.min_weighted_error_rate_threshold` with ``cost=0.5``"
)
.add_prototype("negatives, positives, [is_sorted]", "threshold")
.add_parameter("negatives, positives", "array_like(1D, float)", "The set of negative and positive scores to compute the threshold")
.add_parameter("is_sorted", "bool", "[Default: ``False``] Are both sets of scores already in ascendantly sorted order?")
.add_return("threshold", "float", "The threshold for which the weighted error rate is minimal")
;
static PyObject* min_hter_threshold(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  char** kwlist = min_hter_threshold_doc.kwlist();

  PyBlitzArrayObject* neg;
  PyBlitzArrayObject* pos;
  PyObject* is_sorted = Py_False;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&|O",
        kwlist,
        &double1d_converter, &neg,
        &double1d_converter, &pos,
        &is_sorted
        )) return 0;

  //protects acquired resources through this scope
  auto neg_ = make_safe(neg);
  auto pos_ = make_safe(pos);

  double result = bob::measure::minHterThreshold(
      *PyBlitzArrayCxx_AsBlitz<double,1>(neg),
      *PyBlitzArrayCxx_AsBlitz<double,1>(pos),
      PyObject_IsTrue(is_sorted)
      );

  return Py_BuildValue("d", result);
BOB_CATCH_FUNCTION("min_hter_threshold", 0)
}

static auto precision_recall_doc = bob::extension::FunctionDoc(
  "precision_recall",
  "Calculates the precision and recall (sensitiveness) values given negative and positive scores and a threshold",
  "Precision and recall are computed as:\n\n"
  ".. math::\n\n"
  "   \\mathrm{precision} = \\frac{tp}{tp + fp}\n\n"
  "   \\mathrm{recall} = \\frac{tp}{tp + fn}\n\n"
  "where :math:`tp` are the true positives, :math:`fp` are the false positives and :math:`fn` are the false negatives.\n\n"
  "``positives`` holds the score information for samples that are labeled to belong to a certain class (a.k.a., 'signal' or 'client'). "
  "``negatives`` holds the score information for samples that are labeled **not** to belong to the class (a.k.a., 'noise' or 'impostor'). "
  "For more precise details about how the method considers error rates, see :py:func:`bob.measure.farfrr`."
)
.add_prototype("negatives, positives, threshold", "precision, recall")
.add_parameter("negatives, positives", "array_like(1D, float)", "The set of negative and positive scores to compute the measurements")
.add_parameter("threshold", "float", "The threshold to compute the measures for")
.add_return("precision", "float", "The precision value for the given negatives and positives")
.add_return("recall", "float", "The recall value for the given negatives and positives")
;
static PyObject* precision_recall(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  static char** kwlist = precision_recall_doc.kwlist();

  PyBlitzArrayObject* neg;
  PyBlitzArrayObject* pos;
  double threshold;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&d",
        kwlist,
        &double1d_converter, &neg,
        &double1d_converter, &pos,
        &threshold
        )) return 0;

  //protects acquired resources through this scope
  auto neg_ = make_safe(neg);
  auto pos_ = make_safe(pos);

  auto result = bob::measure::precision_recall(
      *PyBlitzArrayCxx_AsBlitz<double,1>(neg),
      *PyBlitzArrayCxx_AsBlitz<double,1>(pos),
      threshold);

  return Py_BuildValue("dd", result.first, result.second);
BOB_CATCH_FUNCTION("precision_recall", 0)
}

static auto f_score_doc = bob::extension::FunctionDoc(
  "f_score",
  "This method computes the F-score of the accuracy of the classification",
  "The F-score is a weighted mean of precision and recall measurements, see :py:func:`bob.measure.precision_recall`. "
  "It is computed as:\n\n"
  ".. math::\n\n"
  "    \\mathrm{f-score} = (1 + w^2)\\frac{\\mathrm{precision}\\cdot{}\\mathrm{recall}}{w^2\\cdot{}\\mathrm{precision} + \\mathrm{recall}}\n\n"
  "The weight :math:`w` needs to be non-negative real value. "
  "In case the weight parameter is 1 (the default), the F-score is called F1 score and is a harmonic mean between precision and recall values."
)
.add_prototype("negatives, positives, threshold, [weight]", "f_score")
.add_parameter("negatives, positives", "array_like(1D, float)", "The set of negative and positive scores to compute the precision and recall")
.add_parameter("threshold", "float", "The threshold to compute the precision and recall for")
.add_parameter("weight", "float", "[Default: ``1``] The weight :math:`w` between precision and recall")
.add_return("f_score", "float", "The computed f-score for the given scores and the given threshold")
;
static PyObject* f_score(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  static char** kwlist = f_score_doc.kwlist();

  PyBlitzArrayObject* neg;
  PyBlitzArrayObject* pos;
  double threshold;
  double weight = 1.0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&d|d",
        kwlist,
        &double1d_converter, &neg,
        &double1d_converter, &pos,
        &threshold, &weight
        )) return 0;

  //protects acquired resources through this scope
  auto neg_ = make_safe(neg);
  auto pos_ = make_safe(pos);

  auto result = bob::measure::f_score(
      *PyBlitzArrayCxx_AsBlitz<double,1>(neg),
      *PyBlitzArrayCxx_AsBlitz<double,1>(pos),
      threshold, weight);

  return Py_BuildValue("d",result);
BOB_CATCH_FUNCTION("f_score", 0)
}

static auto correctly_classified_negatives_doc = bob::extension::FunctionDoc(
  "correctly_classified_negatives",
  "This method returns an array composed of booleans that pin-point, which negatives where correctly classified for the given threshold",
  "The pseudo-code for this function is:\n\n"
  "  ``foreach (k in negatives) if negatives[k] < threshold: classified[k] = true else: classified[k] = false``"
)
.add_prototype("negatives, threshold", "classified")
.add_parameter("negatives", "array_like(1D, float)", "The scores generated by comparing objects of different classes")
.add_parameter("threshold", "float", "The threshold, for which scores should be considered to be correctly classified")
.add_return("classified", "array_like(1D, bool)", "The decision for each of the ``negatives``")
;
static PyObject* correctly_classified_negatives(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  static char** kwlist = correctly_classified_negatives_doc.kwlist();

  PyBlitzArrayObject* neg;
  double threshold;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&d",
        kwlist,
        &double1d_converter, &neg,
        &threshold
        )) return 0;

  //protects acquired resources through this scope
  auto neg_ = make_safe(neg);

  auto result = bob::measure::correctlyClassifiedNegatives(
      *PyBlitzArrayCxx_AsBlitz<double,1>(neg),
      threshold);

  return PyBlitzArrayCxx_AsNumpy(result);
BOB_CATCH_FUNCTION("correctly_classified_negatives", 0)
}

static auto correctly_classified_positives_doc = bob::extension::FunctionDoc(
  "correctly_classified_positives",
  "This method returns an array composed of booleans that pin-point, which positives where correctly classified for the given threshold",
  "The pseudo-code for this function is:\n\n"
  "  ``foreach (k in positives) if positives[k] >= threshold: classified[k] = true else: classified[k] = false``"
)
.add_prototype("positives, threshold", "classified")
.add_parameter("positives", "array_like(1D, float)", "The scores generated by comparing objects of the same classes")
.add_parameter("threshold", "float", "The threshold, for which scores should be considered to be correctly classified")
.add_return("classified", "array_like(1D, bool)", "The decision for each of the ``positives``")
;
static PyObject* correctly_classified_positives(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  static char** kwlist = correctly_classified_positives_doc.kwlist();

  PyBlitzArrayObject* pos;
  double threshold;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&d",
        kwlist,
        &double1d_converter, &pos,
        &threshold
        )) return 0;

  //protects acquired resources through this scope
  auto pos_ = make_safe(pos);

  auto result = bob::measure::correctlyClassifiedPositives(
      *PyBlitzArrayCxx_AsBlitz<double,1>(pos),
      threshold);

  return PyBlitzArrayCxx_AsNumpy(result);
BOB_CATCH_FUNCTION("correctly_classified_positives", 0)
}

static auto precision_recall_curve_doc = bob::extension::FunctionDoc(
  "precision_recall_curve",
  "Calculates the precision-recall curve given a set of positive and negative scores and a number of desired points" ,
  "The points in which the curve is calculated are distributed uniformly in the range ``[min(negatives, positives), max(negatives, positives)]``"
)
.add_prototype("negatives, positives, n_points", "curve")
.add_parameter("negatives, positives", "array_like(1D, float)", "The set of negative and positive scores to compute the measurements")
.add_parameter("n_points", "int", "The number of thresholds for which precision and recall should be evaluated")
.add_return("curve", "array_like(2D, float)", "2D array of floats that express the X (precision) and Y (recall) coordinates")
;
static PyObject* precision_recall_curve(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  char** kwlist = precision_recall_curve_doc.kwlist();

  PyBlitzArrayObject* neg;
  PyBlitzArrayObject* pos;
  Py_ssize_t n_points;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&n",
        kwlist,
        &double1d_converter, &neg,
        &double1d_converter, &pos,
        &n_points
        )) return 0;

  //protects acquired resources through this scope
  auto neg_ = make_safe(neg);
  auto pos_ = make_safe(pos);

  auto result = bob::measure::precision_recall_curve(
      *PyBlitzArrayCxx_AsBlitz<double,1>(neg),
      *PyBlitzArrayCxx_AsBlitz<double,1>(pos),
      n_points);

  return PyBlitzArrayCxx_AsNumpy(result);
BOB_CATCH_FUNCTION("precision_recall_curve", 0)
}

static auto far_threshold_doc = bob::extension::FunctionDoc(
  "far_threshold",
  "Computes the threshold such that the real FAR is **at least** the requested ``far_value``",
  ".. note::\n\n"
  "   The scores will be sorted internally, requiring the scores to be copied.\n"
  "   To avoid this copy, you can sort the ``negatives`` scores externally in ascendant order, and set the ``is_sorted`` parameter to ``True``"
)
.add_prototype("negatives, positives, [far_value], [is_sorted]", "threshold")
.add_parameter("negatives", "array_like(1D, float)", "The set of negative scores to compute the FAR threshold")
.add_parameter("positives", "array_like(1D, float)", "Ignored, but needs to be specified -- may be given as ``[]``")
.add_parameter("far_value", "float", "[Default: ``0.001``] The FAR value, for which the threshold should be computed")
.add_parameter("is_sorted", "bool", "[Default: ``False``] Set this to ``True`` if the ``negatives`` are already sorted in ascending order. If ``False``, scores will be sorted internally, which will require more memory")
.add_return("threshold", "float", "The threshold such that the real FAR is at least ``far_value``")
;
static PyObject* far_threshold(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  static char** kwlist = far_threshold_doc.kwlist();

  PyBlitzArrayObject* neg;
  PyBlitzArrayObject* pos;
  double far_value = 0.001;
  PyObject* is_sorted = Py_False;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&|dO",
        kwlist,
        &double1d_converter, &neg,
        &double1d_converter, &pos,
        &far_value,
        is_sorted
        )) return 0;

  //protects acquired resources through this scope
  auto neg_ = make_safe(neg);
  auto pos_ = make_safe(pos);

  auto result = bob::measure::farThreshold(
      *PyBlitzArrayCxx_AsBlitz<double,1>(neg),
      *PyBlitzArrayCxx_AsBlitz<double,1>(pos),
      far_value,
      PyObject_IsTrue(is_sorted)
      );

  return Py_BuildValue("d", result);
BOB_CATCH_FUNCTION("far_threshold", 0)
}

static auto frr_threshold_doc = bob::extension::FunctionDoc(
  "frr_threshold",
  "Computes the threshold such that the real FRR is **at least** the requested ``frr_value``",
  ".. note::\n\n"
  "   The scores will be sorted internally, requiring the scores to be copied.\n"
  "   To avoid this copy, you can sort the ``positives`` scores externally in ascendant order, and set the ``is_sorted`` parameter to ``True``"
)
.add_prototype("negatives, positives, [frr_value], [is_sorted]", "threshold")
.add_parameter("negatives", "array_like(1D, float)", "Ignored, but needs to be specified -- may be given as ``[]``")
.add_parameter("positives", "array_like(1D, float)", "The set of positive scores to compute the FRR threshold")
.add_parameter("frr_value", "float", "[Default: ``0.001``] The FRR value, for which the threshold should be computed")
.add_parameter("is_sorted", "bool", "[Default: ``False``] Set this to ``True`` if the ``positives`` are already sorted in ascendant order. If ``False``, scores will be sorted internally, which will require more memory")
.add_return("threshold", "float", "The threshold such that the real FRR is at least ``frr_value``")
;
static PyObject* frr_threshold(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  char** kwlist = frr_threshold_doc.kwlist();

  PyBlitzArrayObject* neg;
  PyBlitzArrayObject* pos;
  double frr_value = 0.001;
  PyObject* is_sorted = Py_False;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&|dO",
        kwlist,
        &double1d_converter, &neg,
        &double1d_converter, &pos,
        &frr_value,
        &is_sorted
        )) return 0;

  //protects acquired resources through this scope
  auto neg_ = make_safe(neg);
  auto pos_ = make_safe(pos);

  auto result = bob::measure::frrThreshold(
      *PyBlitzArrayCxx_AsBlitz<double,1>(neg),
      *PyBlitzArrayCxx_AsBlitz<double,1>(pos),
      frr_value,
      PyObject_IsTrue(is_sorted)
      );

  return Py_BuildValue("d", result);
BOB_CATCH_FUNCTION("frr_threshold", 0)
}

static auto eer_rocch_doc = bob::extension::FunctionDoc(
  "eer_rocch",
  "Calculates the equal-error-rate (EER) given the input data, on the ROC Convex Hull (ROCCH)",
  "It replicates the EER calculation from the Bosaris toolkit (https://sites.google.com/site/bosaristoolkit/)."
)
.add_prototype("negatives, positives", "threshold")
.add_parameter("negatives, positives", "array_like(1D, float)", "The set of negative and positive scores to compute the threshold")
.add_return("threshold", "float", "The threshold for the equal error rate")
;
static PyObject* eer_rocch(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = eer_rocch_doc.kwlist();

  PyBlitzArrayObject* neg;
  PyBlitzArrayObject* pos;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&",
        kwlist,
        &double1d_converter, &neg,
        &double1d_converter, &pos
        )) return 0;

  //protects acquired resources through this scope
  auto neg_ = make_safe(neg);
  auto pos_ = make_safe(pos);

  auto result = bob::measure::eerRocch(
      *PyBlitzArrayCxx_AsBlitz<double,1>(neg),
      *PyBlitzArrayCxx_AsBlitz<double,1>(pos)
      );

  return Py_BuildValue("d", result);
BOB_CATCH_FUNCTION("eer_rocch", 0)
}

static auto rocch_doc = bob::extension::FunctionDoc(
  "rocch",
  "Calculates the ROC Convex Hull (ROCCH) curve given a set of positive and negative scores"
)
.add_prototype("negatives, positives", "curve")
.add_parameter("negatives, positives", "array_like(1D, float)", "The set of negative and positive scores to compute the curve")
.add_return("curve", "array_like(2D, float)", "The ROC curve, with the first row containing the FAR, and the second row containing the FRR")
;
static PyObject* rocch(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = rocch_doc.kwlist();

  PyBlitzArrayObject* neg;
  PyBlitzArrayObject* pos;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&",
        kwlist,
        &double1d_converter, &neg,
        &double1d_converter, &pos
        )) return 0;

  //protects acquired resources through this scope
  auto neg_ = make_safe(neg);
  auto pos_ = make_safe(pos);

  auto result = bob::measure::rocch(
      *PyBlitzArrayCxx_AsBlitz<double,1>(neg),
      *PyBlitzArrayCxx_AsBlitz<double,1>(pos)
      );

  return PyBlitzArrayCxx_AsNumpy(result);
BOB_CATCH_FUNCTION("rocch", 0)
}

static auto rocch2eer_doc = bob::extension::FunctionDoc(
  "rocch2eer",
  "Calculates the threshold that is as close as possible to the equal-error-rate (EER) given the input data"
)
.add_prototype("pmiss_pfa", "threshold")
// I don't know, what the pmiss_pfa parameter is, so I leave out its documentation (a .. todo:: will be generated automatically)
//.add_parameter("pmiss_pfa", "array_like(2D, float)", "???")
.add_return("threshold", "float", "The computed threshold, at which the EER can be obtained")
;
static PyObject* rocch2eer(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  static char** kwlist = rocch2eer_doc.kwlist();

  PyBlitzArrayObject* p;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&",
        kwlist,
        &double2d_converter, &p
        )) return 0;

  auto p_ = make_safe(p);

  auto result = bob::measure::rocch2eer(*PyBlitzArrayCxx_AsBlitz<double,2>(p));

  return Py_BuildValue("d", result);
BOB_CATCH_FUNCTION("rocch2eer", 0)
}

static auto roc_for_far_doc = bob::extension::FunctionDoc(
  "roc_for_far",
  "Calculates the ROC curve for a given set of positive and negative scores and the FAR values, for which the FRR should be computed",
  ".. note::\n\n"
  "   The scores will be sorted internally, requiring the scores to be copied.\n"
  "   To avoid this copy, you can sort both sets of scores externally in ascendant order, and set the ``is_sorted`` parameter to ``True``"
)
.add_prototype("negatives, positives, far_list, [is_sorted]", "curve")
.add_parameter("negatives, positives", "array_like(1D, float)", "The set of negative and positive scores to compute the curve")
.add_parameter("far_list", "array_like(1D, float)", "A list of FAR values, for which the FRR values should be computed")
.add_parameter("is_sorted", "bool", "[Default: ``False``] Set this to ``True`` if both sets of scores are already sorted in ascending order. If ``False``, scores will be sorted internally, which will require more memory")
.add_return("curve", "array_like(2D, float)", "The ROC curve, which holds a copy of the given FAR values in row 0, and the corresponding FRR values in row 1")
;
static PyObject* roc_for_far(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = roc_for_far_doc.kwlist();

  PyBlitzArrayObject* neg;
  PyBlitzArrayObject* pos;
  PyBlitzArrayObject* far;
  PyObject* is_sorted = Py_False;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&O&|O",
        kwlist,
        &double1d_converter, &neg,
        &double1d_converter, &pos,
        &double1d_converter, &far,
        &is_sorted
        )) return 0;

  //protects acquired resources through this scope
  auto neg_ = make_safe(neg);
  auto pos_ = make_safe(pos);
  auto far_ = make_safe(far);

  auto result = bob::measure::roc_for_far(
      *PyBlitzArrayCxx_AsBlitz<double,1>(neg),
      *PyBlitzArrayCxx_AsBlitz<double,1>(pos),
      *PyBlitzArrayCxx_AsBlitz<double,1>(far),
      PyObject_IsTrue(is_sorted)
      );

  return PyBlitzArrayCxx_AsNumpy(result);
BOB_CATCH_FUNCTION("roc_for_far", 0)
}

static PyMethodDef module_methods[] = {
    {
      epc_doc.name(),
      (PyCFunction)epc,
      METH_VARARGS|METH_KEYWORDS,
      epc_doc.doc()
    },
    {
      det_doc.name(),
      (PyCFunction)det,
      METH_VARARGS|METH_KEYWORDS,
      det_doc.doc()
    },
    {
      ppndf_doc.name(),
      (PyCFunction)ppndf,
      METH_VARARGS|METH_KEYWORDS,
      ppndf_doc.doc()
    },
    {
      roc_doc.name(),
      (PyCFunction)roc,
      METH_VARARGS|METH_KEYWORDS,
      roc_doc.doc()
    },
    {
      farfrr_doc.name(),
      (PyCFunction)farfrr,
      METH_VARARGS|METH_KEYWORDS,
      farfrr_doc.doc()
    },
    {
      eer_threshold_doc.name(),
      (PyCFunction)eer_threshold,
      METH_VARARGS|METH_KEYWORDS,
      eer_threshold_doc.doc()
    },
    {
      min_weighted_error_rate_threshold_doc.name(),
      (PyCFunction)min_weighted_error_rate_threshold,
      METH_VARARGS|METH_KEYWORDS,
      min_weighted_error_rate_threshold_doc.doc()
    },
    {
      min_hter_threshold_doc.name(),
      (PyCFunction)min_hter_threshold,
      METH_VARARGS|METH_KEYWORDS,
      min_hter_threshold_doc.doc()
    },
    {
      precision_recall_doc.name(),
      (PyCFunction)precision_recall,
      METH_VARARGS|METH_KEYWORDS,
      precision_recall_doc.doc()
    },
    {
      f_score_doc.name(),
      (PyCFunction)f_score,
      METH_VARARGS|METH_KEYWORDS,
      f_score_doc.doc()
    },
    {
      correctly_classified_negatives_doc.name(),
      (PyCFunction)correctly_classified_negatives,
      METH_VARARGS|METH_KEYWORDS,
      correctly_classified_negatives_doc.doc()
    },
    {
      correctly_classified_positives_doc.name(),
      (PyCFunction)correctly_classified_positives,
      METH_VARARGS|METH_KEYWORDS,
      correctly_classified_positives_doc.doc()
    },
    {
      precision_recall_curve_doc.name(),
      (PyCFunction)precision_recall_curve,
      METH_VARARGS|METH_KEYWORDS,
      precision_recall_curve_doc.doc()
    },
    {
      far_threshold_doc.name(),
      (PyCFunction)far_threshold,
      METH_VARARGS|METH_KEYWORDS,
      far_threshold_doc.doc()
    },
    {
      frr_threshold_doc.name(),
      (PyCFunction)frr_threshold,
      METH_VARARGS|METH_KEYWORDS,
      frr_threshold_doc.doc()
    },
    {
      eer_rocch_doc.name(),
      (PyCFunction)eer_rocch,
      METH_VARARGS|METH_KEYWORDS,
      eer_rocch_doc.doc()
    },
    {
      rocch_doc.name(),
      (PyCFunction)rocch,
      METH_VARARGS|METH_KEYWORDS,
      rocch_doc.doc()
    },
    {
      rocch2eer_doc.name(),
      (PyCFunction)rocch2eer,
      METH_VARARGS|METH_KEYWORDS,
      rocch2eer_doc.doc()
    },
    {
      roc_for_far_doc.name(),
      (PyCFunction)roc_for_far,
      METH_VARARGS|METH_KEYWORDS,
      roc_for_far_doc.doc()
    },
    {0}  /* Sentinel */
};

PyDoc_STRVAR(module_docstr, "Bob metrics and performance figures");

#if PY_VERSION_HEX >= 0x03000000
static PyModuleDef module_definition = {
  PyModuleDef_HEAD_INIT,
  BOB_EXT_MODULE_NAME,
  module_docstr,
  -1,
  module_methods,
  0, 0, 0, 0
};
#endif

static PyObject* create_module (void) {

# if PY_VERSION_HEX >= 0x03000000
  PyObject* m = PyModule_Create(&module_definition);
  auto m_ = make_xsafe(m);
  const char* ret = "O";
# else
  PyObject* m = Py_InitModule3(BOB_EXT_MODULE_NAME, module_methods, module_docstr);
  const char* ret = "N";
# endif
  if (!m) return 0;

  /* imports bob.blitz C-API + dependencies */
  if (import_bob_blitz() < 0) return 0;
  if (import_bob_core_logging() < 0) return 0;
  if (import_bob_io_base() < 0) return 0;

  return Py_BuildValue(ret, m);
}

PyMODINIT_FUNC BOB_EXT_ENTRY_NAME (void) {
# if PY_VERSION_HEX >= 0x03000000
  return
# endif
    create_module();
}
