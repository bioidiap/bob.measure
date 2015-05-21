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

#include "error.h"

static int double1d_converter(PyObject* o, PyBlitzArrayObject** a) {
  if (PyBlitzArray_Converter(o, a) != 0) return 1;
  // in this case, *a is set to a new reference
  if ((*a)->type_num != NPY_FLOAT64 || (*a)->ndim != 1) {
    PyErr_Format(PyExc_TypeError, "cannot convert blitz::Array<%s,%" PY_FORMAT_SIZE_T "d> to a blitz::Array<double,1>", PyBlitzArray_TypenumAsString((*a)->type_num), (*a)->ndim);
    return 1;
  }
  return 0;
}

PyDoc_STRVAR(s_epc_str, "epc");
PyDoc_STRVAR(s_epc_doc,
"epc(dev_negatives, dev_positives, test_negatives, test_positives, n_points) -> numpy.ndarray\n\
\n\
Calculates points of an Expected Performance Curve (EPC).\n\
\n\
Calculates the EPC curve given a set of positive and negative scores\n\
and a desired number of points. Returns a two-dimensional\n\
blitz::Array of doubles that express the X (cost) and Y (HTER on\n\
the test set given the min. HTER threshold on the development set)\n\
coordinates in this order. Please note that, in order to calculate\n\
the EPC curve, one needs two sets of data comprising a development\n\
set and a test set. The minimum weighted error is calculated on the\n\
development set and then applied to the test set to evaluate the\n\
half-total error rate at that position.\n\
\n\
The EPC curve plots the HTER on the test set for various values of\n\
'cost'. For each value of 'cost', a threshold is found that provides\n\
the minimum weighted error (see\n\
:py:func:`bob.measure.min_weighted_error_rate_threshold()`)\n\
on the development set. Each threshold is consecutively applied to\n\
the test set and the resulting HTER values are plotted in the EPC.\n\
\n\
The cost points in which the EPC curve are calculated are\n\
distributed uniformily in the range :math:`[0.0, 1.0]`.\n\
");

static PyObject* epc(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {
    "dev_negatives",
    "dev_positives",
    "test_positives",
    "test_negatives",
    "n_points",
    0 /* Sentinel */
  };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* dev_neg = 0;
  PyBlitzArrayObject* dev_pos = 0;
  PyBlitzArrayObject* test_neg = 0;
  PyBlitzArrayObject* test_pos = 0;
  Py_ssize_t n_points = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&O&O&n",
        kwlist,
        &double1d_converter, &dev_neg,
        &double1d_converter, &dev_pos,
        &double1d_converter, &test_neg,
        &double1d_converter, &test_pos,
        &n_points
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
      n_points);

  return PyBlitzArray_NUMPY_WRAP(PyBlitzArrayCxx_NewFromArray(result));

}

PyDoc_STRVAR(s_det_str, "det");
PyDoc_STRVAR(s_det_doc,
"det(negatives, positives, n_points) -> numpy.ndarray\n\
\n\
Calculates points of an Detection Error-Tradeoff Curve (DET).\n\
\n\
Calculates the DET curve given a set of positive and negative scores and\n\
a desired number of points. Returns a two-dimensional array of doubles\n\
that express on its rows:\n\
\n\
[0]\n\
   X axis values in the normal deviate scale for the false-rejections\n\
\n\
[1]\n\
   Y axis values in the normal deviate scale for the false-accepts\n\
\n\
You can plot the results using your preferred tool to first create a\n\
plot using rows 0 and 1 from the returned value and then replace the\n\
X/Y axis annotation using a pre-determined set of tickmarks as\n\
recommended by NIST. The algorithm that calculates the deviate\n\
scale is based on function ppndf() from the NIST package DETware\n\
version 2.1, freely available on the internet. Please consult it for\n\
more details. By 20.04.2011, you could find such package `here\n\
<http://www.itl.nist.gov/iad/mig/tools/>`_.\n\
");

static PyObject* det(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {
    "negatives",
    "positives",
    "n_points",
    0 /* Sentinel */
  };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* neg = 0;
  PyBlitzArrayObject* pos = 0;
  Py_ssize_t n_points = 0;

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

  return PyBlitzArray_NUMPY_WRAP(PyBlitzArrayCxx_NewFromArray(result));

}

PyDoc_STRVAR(s_ppndf_str, "ppndf");
PyDoc_STRVAR(s_ppndf_doc,
"ppndf(value) -> float\n\
\n\
Returns the Deviate Scale equivalent of a false rejection/acceptance ratio.\n\
\n\
The algorithm that calculates the deviate scale is based on\n\
function ppndf() from the NIST package DETware version 2.1,\n\
freely available on the internet. Please consult it for more\n\
details.\n\
");

static PyObject* ppndf(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {
    "value",
    0 /* Sentinel */
  };
  static char** kwlist = const_cast<char**>(const_kwlist);

  double v = 0.;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "d", kwlist, &v)) return 0;

  return PyFloat_FromDouble(bob::measure::ppndf(v));

}

PyDoc_STRVAR(s_roc_str, "roc");
PyDoc_STRVAR(s_roc_doc,
"roc(negatives, positives, n_points) -> numpy.ndarray\n\
\n\
Calculates points of an Receiver Operating Characteristic (ROC).\n\
\n\
Calculates the ROC curve given a set of positive and negative scores\n\
and a desired number of points. Returns a two-dimensional array\n\
of doubles that express the X (FAR) and Y (FRR) coordinates in this\n\
order. The points in which the ROC curve are calculated are\n\
distributed uniformily in the range [min(negatives, positives),\n\
max(negatives, positives)].\n\
");

static PyObject* roc(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {
    "negatives",
    "positives",
    "n_points",
    0 /* Sentinel */
  };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* neg = 0;
  PyBlitzArrayObject* pos = 0;
  Py_ssize_t n_points = 0;

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

  return PyBlitzArray_NUMPY_WRAP(PyBlitzArrayCxx_NewFromArray(result));

}

PyDoc_STRVAR(s_farfrr_str, "farfrr");
PyDoc_STRVAR(s_farfrr_doc,
"farfrr(negatives, positives, threshold) -> (float, float)\n\
\n\
Calculates the false-acceptance (FA) ratio and the FR false-rejection\n\
(FR) ratio given positive and negative scores and a threshold.\n\
``positives`` holds the score information for samples that are\n\
labelled to belong to a certain class (a.k.a., 'signal' or 'client').\n\
``negatives`` holds the score information for samples that are\n\
labelled **not** to belong to the class (a.k.a., 'noise' or 'impostor').\n\
\n\
It is expected that 'positive' scores are, at least by design, greater\n\
than ``negative`` scores. So, every positive value that falls bellow the\n\
threshold is considered a false-rejection (FR). ``negative`` samples\n\
that fall above the threshold are considered a false-accept (FA).\n\
\n\
Positives that fall on the threshold (exactly) are considered\n\
correctly classified. Negatives that fall on the threshold (exactly)\n\
are considered **incorrectly** classified. This equivalent to setting\n\
the comparision like this pseudo-code:\n\
\n\
foreach (positive as K) if K < threshold: falseRejectionCount += 1\n\
foreach (negative as K) if K >= threshold: falseAcceptCount += 1\n\
\n\
The ``threshold`` value does not necessarily have to fall in the range\n\
covered by the input scores (negatives and positives altogether), but\n\
if it does not, the output will be either (1.0, 0.0) or (0.0, 1.0)\n\
depending on the side the threshold falls.\n\
\n\
The output is in form of a tuple of two double-precision real\n\
numbers. The numbers range from 0 to 1. The first element of the pair\n\
is the false-accept ratio (FAR). The second element of the pair is the\n\
false-rejection ratio (FRR).\n\
\n\
It is possible that scores are inverted in the negative/positive\n\
sense. In some setups the designer may have setup the system so\n\
``positive`` samples have a smaller score than the ``negative`` ones.\n\
In this case, make sure you normalize the scores so positive samples\n\
have greater scores before feeding them into this method.\n\
");

static PyObject* farfrr(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {
    "negatives",
    "positives",
    "threshold",
    0 /* Sentinel */
  };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* neg = 0;
  PyBlitzArrayObject* pos = 0;
  double threshold = 0.;

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

  PyObject* retval = PyTuple_New(2);
  PyTuple_SET_ITEM(retval, 0, PyFloat_FromDouble(result.first));
  PyTuple_SET_ITEM(retval, 1, PyFloat_FromDouble(result.second));
  return retval;

}

PyDoc_STRVAR(s_eer_threshold_str, "eer_threshold");
PyDoc_STRVAR(s_eer_threshold_doc,
"eer_threshold(negatives, positives) -> float\n\
\n\
Calculates the threshold that is as close as possible to the\n\
equal-error-rate (EER) given the input data. The EER should be the\n\
point where the FAR equals the FRR. Graphically, this would be\n\
equivalent to the intersection between the ROC (or DET) curves and the\n\
identity.\n\
");

static PyObject* eer_threshold(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {
    "negatives",
    "positives",
    0 /* Sentinel */
  };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* neg = 0;
  PyBlitzArrayObject* pos = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&",
        kwlist,
        &double1d_converter, &neg,
        &double1d_converter, &pos
        )) return 0;

  //protects acquired resources through this scope
  auto neg_ = make_safe(neg);
  auto pos_ = make_safe(pos);

  double result = bob::measure::eerThreshold(
      *PyBlitzArrayCxx_AsBlitz<double,1>(neg),
      *PyBlitzArrayCxx_AsBlitz<double,1>(pos));

  return PyFloat_FromDouble(result);

}

PyDoc_STRVAR(s_min_weighted_error_rate_threshold_str,
    "min_weighted_error_rate_threshold");
PyDoc_STRVAR(s_min_weighted_error_rate_threshold_doc,
"min_weighted_error_rate_threshold(negatives, positives, cost) -> float\n\
\n\
Calculates the threshold that minimizes the error rate, given the\n\
input data. An optional parameter 'cost' determines the relative\n\
importance between false-accepts and false-rejections. This number\n\
should be between 0 and 1 and will be clipped to those extremes. The\n\
value to minimize becomes: ER_cost = [cost * FAR] + [(1-cost) * FRR].\n\
The higher the cost, the higher the importance given to **not** making\n\
mistakes classifying negatives/noise/impostors.\n\
");

static PyObject* min_weighted_error_rate_threshold(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {
    "negatives",
    "positives",
    "cost",
    0 /* Sentinel */
  };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* neg = 0;
  PyBlitzArrayObject* pos = 0;
  double cost = 0.;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&d",
        kwlist,
        &double1d_converter, &neg,
        &double1d_converter, &pos,
        &cost
        )) return 0;

  //protects acquired resources through this scope
  auto neg_ = make_safe(neg);
  auto pos_ = make_safe(pos);

  double result = bob::measure::minWeightedErrorRateThreshold(
      *PyBlitzArrayCxx_AsBlitz<double,1>(neg),
      *PyBlitzArrayCxx_AsBlitz<double,1>(pos),
      cost);

  return PyFloat_FromDouble(result);

}

PyDoc_STRVAR(s_min_hter_threshold_str, "min_hter_threshold");
PyDoc_STRVAR(s_min_hter_threshold_doc,
"min_hter_threshold(negatives, positives) -> float\n\
\n\
Calculates the :py:func:`min_weighted_error_rate_threshold()` with\n\
the cost set to 0.5.\n\
");

static PyObject* min_hter_threshold(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {
    "negatives",
    "positives",
    0 /* Sentinel */
  };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* neg = 0;
  PyBlitzArrayObject* pos = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&",
        kwlist,
        &double1d_converter, &neg,
        &double1d_converter, &pos
        )) return 0;

  //protects acquired resources through this scope
  auto neg_ = make_safe(neg);
  auto pos_ = make_safe(pos);

  double result = bob::measure::minHterThreshold(
      *PyBlitzArrayCxx_AsBlitz<double,1>(neg),
      *PyBlitzArrayCxx_AsBlitz<double,1>(pos));

  return PyFloat_FromDouble(result);

}

PyDoc_STRVAR(s_precision_recall_str, "precision_recall");
PyDoc_STRVAR(s_precision_recall_doc,
"precision_recall(negatives, positives, threshold) -> (float, float)\n\
\n\
Calculates the precision and recall (sensitiveness) values given\n\
positive and negative scores and a threshold. ``positives`` holds the\n\
score information for samples that are labeled to belong to a certain\n\
class (a.k.a., 'signal' or 'client'). ``negatives`` holds the score\n\
information for samples that are labeled **not** to belong to the class\n\
(a.k.a., 'noise' or 'impostor'). For more precise details about how\n\
the method considers error rates, please refer to the documentation of\n\
the method :py:func:`farfrr`.\n\
");

static PyObject* precision_recall(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {
    "negatives",
    "positives",
    "threshold",
    0 /* Sentinel */
  };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* neg = 0;
  PyBlitzArrayObject* pos = 0;
  double threshold = 0.;

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

  PyObject* retval = PyTuple_New(2);
  PyTuple_SET_ITEM(retval, 0, PyFloat_FromDouble(result.first));
  PyTuple_SET_ITEM(retval, 1, PyFloat_FromDouble(result.second));
  return retval;

}

PyDoc_STRVAR(s_f_score_str, "f_score");
PyDoc_STRVAR(s_f_score_doc,
"f_score(negatives, positives, threshold[, weight=1.0]) -> float\n\
\n\
This method computes F-score of the accuracy of the classification. It\n\
is a weighted mean of precision and recall measurements. The weight\n\
parameter needs to be non-negative real value. In case the weight\n\
parameter is 1, the F-score is called F1 score and is a harmonic mean\n\
between precision and recall values.\n\
");

static PyObject* f_score(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {
    "negatives",
    "positives",
    "threshold",
    "weight",
    0 /* Sentinel */
  };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* neg = 0;
  PyBlitzArrayObject* pos = 0;
  double threshold = 0.;
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

  return PyFloat_FromDouble(result);

}

PyDoc_STRVAR(s_correctly_classified_negatives_str, "correctly_classified_negatives");
PyDoc_STRVAR(s_correctly_classified_negatives_doc,
"correctly_classified_negatives(negatives, threshold) -> int\n\
\n\
This method returns an array composed of booleans that pin-point\n\
which negatives where correctly classified in a \"negative\" score\n\
sample, given a threshold. It runs the formula: foreach (element k in\n\
negative) if negative[k] < threshold: returnValue[k] = true else:\n\
returnValue[k] = false\n\
");

static PyObject* correctly_classified_negatives(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {
    "negatives",
    "threshold",
    0 /* Sentinel */
  };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* neg = 0;
  double threshold = 0.;

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

  return PyBlitzArray_NUMPY_WRAP(PyBlitzArrayCxx_NewFromArray(result));

}

PyDoc_STRVAR(s_correctly_classified_positives_str, "correctly_classified_positives");
PyDoc_STRVAR(s_correctly_classified_positives_doc,
"correctly_classified_positives(positives, threshold) -> numpy.ndarray\n\
\n\
This method returns a 1D array composed of booleans that pin-point\n\
which positives where correctly classified in a 'positive' score\n\
sample, given a threshold. It runs the formula: foreach (element k in\n\
positive) if positive[k] >= threshold: returnValue[k] = true else:\n\
returnValue[k] = false\n\
");

static PyObject* correctly_classified_positives(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {
    "positives",
    "threshold",
    0 /* Sentinel */
  };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* pos = 0;
  double threshold = 0.;

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

  return PyBlitzArray_NUMPY_WRAP(PyBlitzArrayCxx_NewFromArray(result));

}

PyDoc_STRVAR(s_precision_recall_curve_str, "precision_recall_curve");
PyDoc_STRVAR(s_precision_recall_curve_doc,
"precision_recall_curve(negatives, positives, n_points) -> numpy.ndarray\n\
\n\
Calculates the precision-recall curve given a set of positive and\n\
negative scores and a number of desired points. Returns a\n\
two-dimensional array of doubles that express the X (precision)\n\
and Y (recall) coordinates in this order. The points in which the\n\
curve is calculated are distributed uniformly in\n\
the range [min(negatives, positives), max(negatives, positives)].\n\
");

static PyObject* precision_recall_curve(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {
    "negatives",
    "positives",
    "n_points",
    0 /* Sentinel */
  };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* neg = 0;
  PyBlitzArrayObject* pos = 0;
  Py_ssize_t n_points = 0;

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

  return PyBlitzArray_NUMPY_WRAP(PyBlitzArrayCxx_NewFromArray(result));

}

PyDoc_STRVAR(s_far_threshold_str, "far_threshold");
PyDoc_STRVAR(s_far_threshold_doc,
"far_threshold(negatives, positives[, far_value=0.001]) -> float\n\
\n\
Computes the threshold such that the real FAR is *at least* the\n\
requested ``far_value``.\n\
\n\
Keyword parameters:\n\
\n\
negatives\n\
   The impostor scores to be used for computing the FAR\n\
\n\
positives\n\
   The client scores; ignored by this function\n\
\n\
far_value\n\
   The FAR value where the threshold should be computed\n\
\n\
Returns the computed threshold (float)\n\
");

static PyObject* far_threshold(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {
    "negatives",
    "positives",
    "threshold",
    0 /* Sentinel */
  };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* neg = 0;
  PyBlitzArrayObject* pos = 0;
  double far_value = 0.001;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&|d",
        kwlist,
        &double1d_converter, &neg,
        &double1d_converter, &pos,
        &far_value
        )) return 0;

  //protects acquired resources through this scope
  auto neg_ = make_safe(neg);
  auto pos_ = make_safe(pos);

  auto result = bob::measure::farThreshold(
      *PyBlitzArrayCxx_AsBlitz<double,1>(neg),
      *PyBlitzArrayCxx_AsBlitz<double,1>(pos),
      far_value);

  return PyFloat_FromDouble(result);

}

PyDoc_STRVAR(s_frr_threshold_str, "frr_threshold");
PyDoc_STRVAR(s_frr_threshold_doc,
"frr_threshold(negatives, positives[, frr_value=0.001]) -> float\n\
\n\
Computes the threshold such that the real FRR is *at least* the\n\
requested ``frr_value``.\n\
\n\
Keyword parameters:\n\
\n\
negatives\n\
   The impostor scores; ignored by this function\n\
\n\
positives\n\
   The client scores to be used for computing the FRR\n\
\n\
frr_value\n\
   The FRR value where the threshold should be computed\n\
\n\
Returns the computed threshold (float)\n\
");

static PyObject* frr_threshold(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {
    "negatives",
    "positives",
    "frr_value",
    0 /* Sentinel */
  };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* neg = 0;
  PyBlitzArrayObject* pos = 0;
  double frr_value = 0.001;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&|d",
        kwlist,
        &double1d_converter, &neg,
        &double1d_converter, &pos,
        &frr_value
        )) return 0;

  //protects acquired resources through this scope
  auto neg_ = make_safe(neg);
  auto pos_ = make_safe(pos);

  auto result = bob::measure::frrThreshold(
      *PyBlitzArrayCxx_AsBlitz<double,1>(neg),
      *PyBlitzArrayCxx_AsBlitz<double,1>(pos),
      frr_value);

  return PyFloat_FromDouble(result);

}

PyDoc_STRVAR(s_eer_rocch_str, "eer_rocch");
PyDoc_STRVAR(s_eer_rocch_doc,
"eer_rocch(negatives, positives) -> float\n\
\n\
Calculates the equal-error-rate (EER) given the input data, on\n\
the ROC Convex Hull as done in the Bosaris toolkit\n\
(https://sites.google.com/site/bosaristoolkit/).\n\
");

static PyObject* eer_rocch(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {
    "negatives",
    "positives",
    0 /* Sentinel */
  };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* neg = 0;
  PyBlitzArrayObject* pos = 0;

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

  return PyFloat_FromDouble(result);

}

PyDoc_STRVAR(s_rocch_str, "rocch");
PyDoc_STRVAR(s_rocch_doc,
"rocch(negatives, positives) -> numpy.ndarray\n\
\n\
Calculates the ROC Convex Hull curve given a set of positive and\n\
negative scores. Returns a two-dimensional array of doubles\n\
that express the X (FAR) and Y (FRR) coordinates in this order.\n\
");

static PyObject* rocch(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {
    "negatives",
    "positives",
    0 /* Sentinel */
  };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* neg = 0;
  PyBlitzArrayObject* pos = 0;

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

  return PyBlitzArray_NUMPY_WRAP(PyBlitzArrayCxx_NewFromArray(result));

}

static int double2d_converter(PyObject* o, PyBlitzArrayObject** a) {
  if (PyBlitzArray_Converter(o, a) != 0) return 1;
  // in this case, *a is set to a new reference
  if ((*a)->type_num != NPY_FLOAT64 || (*a)->ndim != 2) {
    PyErr_Format(PyExc_TypeError, "cannot convert blitz::Array<%s,%" PY_FORMAT_SIZE_T "d> to a blitz::Array<double,2>", PyBlitzArray_TypenumAsString((*a)->type_num), (*a)->ndim);
    return 1;
  }
  return 0;
}

PyDoc_STRVAR(s_rocch2eer_str, "rocch2eer");
PyDoc_STRVAR(s_rocch2eer_doc,
"rocch2eer(pmiss_pfa) -> float\n\
\n\
Calculates the threshold that is as close as possible to the\n\
equal-error-rate (EER) given the input data.\n\
");

static PyObject* rocch2eer(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {
    "pmiss_pfa",
    0 /* Sentinel */
  };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* p = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&",
        kwlist,
        &double2d_converter, &p
        )) return 0;

  auto p_ = make_safe(p);

  auto result = bob::measure::rocch2eer(*PyBlitzArrayCxx_AsBlitz<double,2>(p));

  return PyFloat_FromDouble(result);

}

PyDoc_STRVAR(s_roc_for_far_str, "roc_for_far");
PyDoc_STRVAR(s_roc_for_far_doc,
"roc_for_far(negatives, positives, far_list) -> numpy.ndarray\n\
\n\
Calculates the ROC curve given a set of positive and negative\n\
scores and the FAR values for which the FRR should be computed.\n\
The resulting ROC curve holds a copy of the given FAR values (row\n\
0), and the corresponding FRR values (row 1).\n\
");

static PyObject* roc_for_far(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {
    "negatives",
    "positives",
    "far_list",
    0 /* Sentinel */
  };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* neg = 0;
  PyBlitzArrayObject* pos = 0;
  PyBlitzArrayObject* list = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&O&",
        kwlist,
        &double1d_converter, &neg,
        &double1d_converter, &pos,
        &double1d_converter, &list
        )) return 0;

  //protects acquired resources through this scope
  auto neg_ = make_safe(neg);
  auto pos_ = make_safe(pos);
  auto list_ = make_safe(list);

  auto result = bob::measure::roc_for_far(
      *PyBlitzArrayCxx_AsBlitz<double,1>(neg),
      *PyBlitzArrayCxx_AsBlitz<double,1>(pos),
      *PyBlitzArrayCxx_AsBlitz<double,1>(list)
      );

  return PyBlitzArray_NUMPY_WRAP(PyBlitzArrayCxx_NewFromArray(result));

}

static PyMethodDef module_methods[] = {
    {
      s_epc_str,
      (PyCFunction)epc,
      METH_VARARGS|METH_KEYWORDS,
      s_epc_doc
    },
    {
      s_det_str,
      (PyCFunction)det,
      METH_VARARGS|METH_KEYWORDS,
      s_det_doc
    },
    {
      s_ppndf_str,
      (PyCFunction)ppndf,
      METH_VARARGS|METH_KEYWORDS,
      s_ppndf_doc
    },
    {
      s_roc_str,
      (PyCFunction)roc,
      METH_VARARGS|METH_KEYWORDS,
      s_roc_doc
    },
    {
      s_farfrr_str,
      (PyCFunction)farfrr,
      METH_VARARGS|METH_KEYWORDS,
      s_farfrr_doc
    },
    {
      s_eer_threshold_str,
      (PyCFunction)eer_threshold,
      METH_VARARGS|METH_KEYWORDS,
      s_eer_threshold_doc
    },
    {
      s_min_weighted_error_rate_threshold_str,
      (PyCFunction)min_weighted_error_rate_threshold,
      METH_VARARGS|METH_KEYWORDS,
      s_min_weighted_error_rate_threshold_doc
    },
    {
      s_min_hter_threshold_str,
      (PyCFunction)min_hter_threshold,
      METH_VARARGS|METH_KEYWORDS,
      s_min_hter_threshold_doc
    },
    {
      s_precision_recall_str,
      (PyCFunction)precision_recall,
      METH_VARARGS|METH_KEYWORDS,
      s_precision_recall_doc
    },
    {
      s_f_score_str,
      (PyCFunction)f_score,
      METH_VARARGS|METH_KEYWORDS,
      s_f_score_doc
    },
    {
      s_correctly_classified_negatives_str,
      (PyCFunction)correctly_classified_negatives,
      METH_VARARGS|METH_KEYWORDS,
      s_correctly_classified_negatives_doc
    },
    {
      s_correctly_classified_positives_str,
      (PyCFunction)correctly_classified_positives,
      METH_VARARGS|METH_KEYWORDS,
      s_correctly_classified_positives_doc
    },
    {
      s_precision_recall_curve_str,
      (PyCFunction)precision_recall_curve,
      METH_VARARGS|METH_KEYWORDS,
      s_precision_recall_curve_doc
    },
    {
      s_far_threshold_str,
      (PyCFunction)far_threshold,
      METH_VARARGS|METH_KEYWORDS,
      s_far_threshold_doc
    },
    {
      s_frr_threshold_str,
      (PyCFunction)frr_threshold,
      METH_VARARGS|METH_KEYWORDS,
      s_frr_threshold_doc
    },
    {
      s_eer_rocch_str,
      (PyCFunction)eer_rocch,
      METH_VARARGS|METH_KEYWORDS,
      s_eer_rocch_doc
    },
    {
      s_rocch_str,
      (PyCFunction)rocch,
      METH_VARARGS|METH_KEYWORDS,
      s_rocch_doc
    },
    {
      s_rocch2eer_str,
      (PyCFunction)rocch2eer,
      METH_VARARGS|METH_KEYWORDS,
      s_rocch2eer_doc
    },
    {
      s_roc_for_far_str,
      (PyCFunction)roc_for_far,
      METH_VARARGS|METH_KEYWORDS,
      s_roc_for_far_doc
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
# else
  PyObject* m = Py_InitModule3(BOB_EXT_MODULE_NAME, module_methods, module_docstr);
# endif
  if (!m) return 0;
  auto m_ = make_safe(m); ///< protects against early returns

  /* imports bob.blitz C-API + dependencies */
  if (import_bob_blitz() < 0) return 0;
  if (import_bob_core_logging() < 0) return 0;
  if (import_bob_io_base() < 0) return 0;

  return Py_BuildValue("O", m);
}

PyMODINIT_FUNC BOB_EXT_ENTRY_NAME (void) {
# if PY_VERSION_HEX >= 0x03000000
  return
# endif
    create_module();
}
