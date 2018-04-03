.. vim: set fileencoding=utf-8 :
.. Sat 16 Nov 20:52:58 2013

============
 Python API
============

This section includes information for using the Python API of
:py:mod:`bob.measure`.


Measurement
-----------

Classification
++++++++++++++

.. autosummary::
   bob.measure.correctly_classified_negatives
   bob.measure.correctly_classified_positives

Single point measurements
+++++++++++++++++++++++++

.. autosummary::
   bob.measure.farfrr
   bob.measure.f_score
   bob.measure.precision_recall
   bob.measure.recognition_rate
   bob.measure.detection_identification_rate
   bob.measure.false_alarm_rate
   bob.measure.eer_rocch

Thresholds
++++++++++

.. autosummary::
   bob.measure.eer_threshold
   bob.measure.rocch2eer
   bob.measure.min_hter_threshold
   bob.measure.min_weighted_error_rate_threshold
   bob.measure.far_threshold
   bob.measure.frr_threshold

Curves
++++++

.. autosummary::
   bob.measure.roc
   bob.measure.rocch
   bob.measure.roc_for_far
   bob.measure.det
   bob.measure.epc
   bob.measure.precision_recall_curve
   bob.measure.cmc

Generic
+++++++

.. autosummary::
   bob.measure.ppndf
   bob.measure.relevance
   bob.measure.mse
   bob.measure.rmse
   bob.measure.get_config


Confidence interval
-------------------

.. autosummary::
   bob.measure.utils.confidence_for_indicator_variable

Calibration
-----------

.. autosummary::
   bob.measure.calibration.cllr
   bob.measure.calibration.min_cllr

Plotting
--------

.. autosummary::
   bob.measure.plot.roc
   bob.measure.plot.det
   bob.measure.plot.det_axis
   bob.measure.plot.epc
   bob.measure.plot.precision_recall_curve
   bob.measure.plot.cmc
   bob.measure.plot.detection_identification_curve

Loading
-------

.. autosummary::
   bob.measure.load.split


Details
-------

.. automodule:: bob.measure
.. automodule:: bob.measure.calibration
.. automodule:: bob.measure.plot
.. automodule:: bob.measure.load
.. automodule:: bob.measure.utils
