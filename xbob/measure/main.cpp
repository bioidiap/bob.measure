/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri 25 Oct 16:54:55 2013
 *
 * @brief Bindings to bob::io
 */

#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif
#include <xbob.blitz/cppapi.h>
#include <bob/measure/error.h>

/**
static tuple farfrr(
    bob::python::const_ndarray negatives,
    bob::python::const_ndarray positives,
    double threshold
){
  std::pair<double, double> retval = bob::measure::farfrr(negatives.cast<double,1>(), positives.cast<double,1>(), threshold);
  return make_tuple(retval.first, retval.second);
}

static tuple precision_recall(
    bob::python::const_ndarray negatives,
    bob::python::const_ndarray positives,
    double threshold
){
  std::pair<double, double> retval = bob::measure::precision_recall(negatives.cast<double,1>(), positives.cast<double,1>(), threshold);
  return make_tuple(retval.first, retval.second);
}
**/

/**
void bind_measure_error() {
  def(
    "farfrr",
    &farfrr,
    (arg("negatives"), arg("positives"), arg("threshold")),
    "Calculates the FA ratio and the FR ratio given positive and negative scores and a threshold. 'positives' holds the score information for samples that are labelled to belong to a certain class (a.k.a., 'signal' or 'client'). 'negatives' holds the score information for samples that are labelled *not* to belong to the class (a.k.a., 'noise' or 'impostor').\n\nIt is expected that 'positive' scores are, at least by design, greater than 'negative' scores. So, every positive value that falls bellow the threshold is considered a false-rejection (FR). 'negative' samples that fall above the threshold are considered a false-accept (FA).\n\nPositives that fall on the threshold (exactly) are considered correctly classified. Negatives that fall on the threshold (exactly) are considered *incorrectly* classified. This equivalent to setting the comparision like this pseudo-code:\n\nforeach (positive as K) if K < threshold: falseRejectionCount += 1\nforeach (negative as K) if K >= threshold: falseAcceptCount += 1\n\nThe 'threshold' value does not necessarily have to fall in the range covered by the input scores (negatives and positives altogether), but if it does not, the output will be either (1.0, 0.0) or (0.0, 1.0) depending on the side the threshold falls.\n\nThe output is in form of a std::pair of two double-precision real numbers. The numbers range from 0 to 1. The first element of the pair is the false-accept ratio. The second element of the pair is the false-rejection ratio.\n\nIt is possible that scores are inverted in the negative/positive sense. In some setups the designer may have setup the system so 'positive' samples have a smaller score than the 'negative' ones. In this case, make sure you normalize the scores so positive samples have greater scores before feeding them into this method."
  );
  
  def(
    "precision_recall",
    &precision_recall,
    (arg("negatives"), arg("positives"), arg("threshold")),
    "Calculates the precision and recall (sensitiveness) values given positive and negative scores and a threshold. 'positives' holds the score information for samples that are labeled to belong to a certain class (a.k.a., 'signal' or 'client'). 'negatives' holds the score information for samples that are labeled *not* to belong to the class (a.k.a., 'noise' or 'impostor'). For more precise details about how the method considers error rates, please refer to the documentation of the method bob.measure.farfrr."
  );

  def(
    "f_score",
    &f_score,
    (arg("negatives"), arg("positives"), arg("threshold"), arg("weight")=1.0),
    "This method computes F score of the accuracy of the classification. It is a weighted mean of precision and recall measurements. The weight parameter needs to be non-negative real value. In case the weight parameter is 1, the F-score is called F1 score and is a harmonic mean between precision and recall values."
  );

  def(
    "correctly_classified_positives",
    &bob_correctly_classified_positives,
    (arg("positives"), arg("threshold")),
    "This method returns a blitz::Array composed of booleans that pin-point which positives where correctly classified in a 'positive' score sample, given a threshold. It runs the formula: foreach (element k in positive) if positive[k] >= threshold: returnValue[k] = true else: returnValue[k] = false"
  );

  def(
    "correctly_classified_negatives",
    &bob_correctly_classified_negatives,
    (arg("negatives"), arg("threshold")),
    "This method returns a blitz::Array composed of booleans that pin-point which negatives where correctly classified in a 'negative' score sample, given a threshold. It runs the formula: foreach (element k in negative) if negative[k] < threshold: returnValue[k] = true else: returnValue[k] = false"
  );

  def(
    "eer_threshold",
    &bob_eer_threshold,
    (arg("negatives"), arg("positives")),
    "Calculates the threshold that is as close as possible to the equal-error-rate (EER) given the input data. The EER should be the point where the FAR equals the FRR. Graphically, this would be equivalent to the intersection between the ROC (or DET) curves and the identity."
  );

 def(
    "eer_rocch",
    &bob_eer_rocch,
    (arg("negatives"), arg("positives")),
    "Calculates the equal-error-rate (EER) given the input data, on the ROC Convex Hull as done in the Bosaris toolkit (https://sites.google.com/site/bosaristoolkit/)."
  );

  def(
    "min_weighted_error_rate_threshold",
    &bob_min_weighted_error_rate_threshold,
    (arg("negatives"), arg("positives"), arg("cost")),
    "Calculates the threshold that minimizes the error rate, given the input data. An optional parameter 'cost' determines the relative importance between false-accepts and false-rejections. This number should be between 0 and 1 and will be clipped to those extremes. The value to minimize becomes: ER_cost = [cost * FAR] + [(1-cost) * FRR]. The higher the cost, the higher the importance given to *not* making mistakes classifying negatives/noise/impostors."
  );

  def(
    "min_hter_threshold",
    &bob_min_hter_threshold,
    (arg("negatives"), arg("positives")),
    "Calculates the min_weighted_error_rate_threshold() when the cost is 0.5."
  );

  def(
    "far_threshold",
    &bob_far_threshold,
    bob_far_threshold_overloads(
      (arg("negatives"), arg("positives"), arg("far_value")=0.001),
      "Computes the threshold such that the real FAR is *at least* the requested ``far_value``.\n\nKeyword parameters:\n\nnegatives\n  The impostor scores to be used for computing the FAR\n\npositives\n  The client scores; ignored by this function\n\nfar_value\n  The FAR value where the threshold should be computed\n\nReturns the computed threshold (float)"
      )
  );

  def(
    "frr_threshold",
    &bob_frr_threshold,
    bob_frr_threshold_overloads(
      (arg("negatives"), arg("positives"), arg("frr_value")=0.001),
      "Computes the threshold such that the real FRR is *at least* the requested ``frr_value``.\n\nKeyword parameters:\n\nnegatives\n  The impostor scores; ignored by this function\n\npositives\n  The client scores to be used for computing the FRR\n\nfrr_value\n\n  The FRR value where the threshold should be computed\n\nReturns the computed threshold (float)"
      )
  );

  def(
    "roc",
    &bob_roc,
    (arg("negatives"), arg("positives"), arg("n_points")),
    "Calculates the ROC curve given a set of positive and negative scores and a desired number of points. Returns a two-dimensional blitz::Array of doubles that express the X (FRR) and Y (FAR) coordinates in this order. The points in which the ROC curve are calculated are distributed uniformily in the range [min(negatives, positives), max(negatives, positives)]."
  );

  def(
    "precision_recall_curve",
    &bob_precision_recall_curve,
    (arg("negatives"), arg("positives"), arg("n_points")),
    "Calculates the precision-recall curve given a set of positive and negative scores and a number of desired points. Returns a two-dimensional blitz::Array of doubles that express the X (precision) and Y (recall) coordinates in this order. The points in which the curve is calculated are distributed uniformly in the range [min(negatives, positives), max(negatives, positives)]."
  );

  def(
    "rocch",
    &bob_rocch,
    (arg("negatives"), arg("positives")),
    "Calculates the ROC Convex Hull curve given a set of positive and negative scores. Returns a two-dimensional blitz::Array of doubles that express the X (FRR) and Y (FAR) coordinates in this order."
  );

  def(
    "rocch2eer",
    &bob_rocch2eer,
    (arg("pmiss_pfa")),
    "Calculates the threshold that is as close as possible to the equal-error-rate (EER) given the input data."
  );

  def(
    "roc_for_far",
    &bob_roc_for_far,
    (arg("negatives"), arg("positives"), arg("far_list")),
    "Calculates the ROC curve given a set of positive and negative scores and the FAR values for which the CAR should be computed. The resulting ROC curve holds a copy of the given FAR values (row 0), and the corresponding FRR values (row 1)."
  );

  def(
    "ppndf",
    &bob::measure::ppndf,
    (arg("value")),
    "Returns the Deviate Scale equivalent of a false rejection/acceptance ratio.\n\nThe algorithm that calculates the deviate scale is based on function ppndf() from the NIST package DETware version 2.1, freely available on the internet. Please consult it for more details."
  );

  def(
    "det",
    &bob_det,
    (arg("negatives"), arg("positives"), arg("n_points")),
    "Calculates the DET curve given a set of positive and negative scores and a desired number of points. Returns a two-dimensional blitz::Array of doubles that express on its rows:\n\n0. X axis values in the normal deviate scale for the false-rejections\n1. Y axis values in the normal deviate scale for the false-accepts\n\nYou can plot the results using your preferred tool to first create a plot using rows 0 and 1 from the returned value and then replace the X/Y axis annotation using a pre-determined set of tickmarks as recommended by NIST.\n\nThe algorithm that calculates the deviate scale is based on function ppndf() from the NIST package DETware version 2.1, freely available on the internet. Please consult it for more details.\n\nBy 20.04.2011, you could find such package here: http://www.itl.nist.gov/iad/mig/tools/"
  );

}
**/

static int double1d_converter(PyObject* o, PyBlitzArrayObject** a) {
  if (PyBlitzArray_Converter(o, a) != 0) return 1;
  // in this case, *a is set to a new reference
  if ((*a)->type_num != NPY_FLOAT64 || (*a)->ndim != 1) {
    PyErr_Format(PyExc_TypeError, "cannot convert blitz::Array<%s,%" PY_FORMAT_SIZE_T "d> to a blitz::Array<double,1>", PyBlitzArray_TypenumAsString((*a)->type_num), (*a)->ndim);
    return 1;
  }
  return 0;
}

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

  blitz::Array<double,2> retval = bob::measure::epc(
      *PyBlitzArrayCxx_AsBlitz<double,1>(dev_neg),
      *PyBlitzArrayCxx_AsBlitz<double,1>(dev_pos),
      *PyBlitzArrayCxx_AsBlitz<double,1>(test_neg),
      *PyBlitzArrayCxx_AsBlitz<double,1>(test_pos),
      n_points);

  PyObject* pyret = reinterpret_cast<PyObject*>(PyBlitzArrayCxx_NewFromArray(retval));

  Py_DECREF(dev_neg);
  Py_DECREF(dev_pos);
  Py_DECREF(test_neg);
  Py_DECREF(test_pos);

  return pyret;

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
:py:func:`xbob.measure.min_weighted_error_rate_threshold()`)\n\
on the development set. Each threshold is consecutively applied to\n\
the test set and the resulting HTER values are plotted in the EPC.\n\
\n\
The cost points in which the EPC curve are calculated are\n\
distributed uniformily in the range :math:`[0.0, 1.0]`.\n\
");

static PyMethodDef library_methods[] = {
    {
      s_epc_str,
      (PyCFunction)epc,
      METH_VARARGS|METH_KEYWORDS,
      s_epc_doc
    },
    {0}  /* Sentinel */
};

PyMODINIT_FUNC XBOB_EXT_ENTRY_NAME (void) {

  PyObject* m = Py_InitModule3(XBOB_EXT_MODULE_NAME, 
      library_methods, "bob::measure bindings");
  PyModule_AddStringConstant(m, "__version__", XBOB_EXT_MODULE_VERSION);

  /* imports the NumPy C-API */
  import_array();

  /* imports xbob.blitz C-API */
  import_xbob_blitz();

}
