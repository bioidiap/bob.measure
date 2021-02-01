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
   bob.measure.eer
   bob.measure.fprfnr
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
   bob.measure.roc_auc_score
   bob.measure.rocch
   bob.measure.roc_for_far
   bob.measure.det
   bob.measure.epc
   bob.measure.precision_recall_curve
   bob.measure.cmc

Figures
-------

.. autosummary::
   bob.measure.script.figure.MeasureBase
   bob.measure.script.figure.Metrics
   bob.measure.script.figure.PlotBase
   bob.measure.script.figure.Roc
   bob.measure.script.figure.Det
   bob.measure.script.figure.Epc
   bob.measure.script.figure.Hist

Generic
+++++++

.. autosummary::
   bob.measure.ppndf
   bob.measure.relevance
   bob.measure.mse
   bob.measure.rmse


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

Utilities
---------

.. autosummary::
   bob.measure.utils.remove_nan
   bob.measure.utils.get_fta
   bob.measure.utils.get_fta_list
   bob.measure.utils.get_thres
   bob.measure.utils.get_colors

CLI options
-----------

.. autosummary::
   bob.measure.script.common_options.scores_argument
   bob.measure.script.common_options.eval_option
   bob.measure.script.common_options.sep_dev_eval_option
   bob.measure.script.common_options.cmc_option
   bob.measure.script.common_options.print_filenames_option
   bob.measure.script.common_options.const_layout_option
   bob.measure.script.common_options.axes_val_option
   bob.measure.script.common_options.thresholds_option
   bob.measure.script.common_options.lines_at_option
   bob.measure.script.common_options.x_rotation_option
   bob.measure.script.common_options.cost_option
   bob.measure.script.common_options.points_curve_option
   bob.measure.script.common_options.n_bins_option
   bob.measure.script.common_options.table_option
   bob.measure.script.common_options.output_plot_file_option
   bob.measure.script.common_options.output_log_metric_option
   bob.measure.script.common_options.criterion_option
   bob.measure.script.common_options.far_option
   bob.measure.script.common_options.figsize_option
   bob.measure.script.common_options.legend_ncols_option
   bob.measure.script.common_options.legend_loc_option
   bob.measure.script.common_options.line_width_option
   bob.measure.script.common_options.marker_style_option
   bob.measure.script.common_options.legends_option
   bob.measure.script.common_options.title_option
   bob.measure.script.common_options.x_label_option
   bob.measure.script.common_options.y_label_option
   bob.measure.script.common_options.style_option
   bob.measure.script.common_options.subplot_option
   bob.measure.script.common_options.legend_ncols_option
   bob.measure.script.common_options.no_legend_option

Details
-------

.. automodule:: bob.measure
.. automodule:: bob.measure.calibration
.. automodule:: bob.measure.plot
.. automodule:: bob.measure.load
.. automodule:: bob.measure.utils
.. automodule:: bob.measure.script.figure
.. automodule:: bob.measure.script.commands
.. automodule:: bob.measure.script.gen
.. automodule:: bob.measure.script.common_options
