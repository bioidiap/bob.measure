""" Click commands for ``bob.measure`` """

from .. import load
from . import common_options, figure

SCORE_FORMAT = (
    "The command takes as input generic 2-column data format as "
    "specified in the documentation of "
    ":py:func:`bob.measure.load.split`."
)
CRITERIA = ("eer", "min-hter", "far")


@common_options.metrics_command(
    common_options.METRICS_HELP.format(
        names="FPR, FNR, precision, recall, F1-score, AUC ROC",
        criteria=CRITERIA,
        score_format=SCORE_FORMAT,
        hter_note=" ",
        command="bob measure metrics",
    ),
    criteria=CRITERIA,
    far_name="FPR",
)
def metrics(ctx, scores, evaluation, **kwargs):
    process = figure.Metrics(ctx, scores, evaluation, load.split)
    process.run()


@common_options.roc_command(
    common_options.ROC_HELP.format(
        score_format=SCORE_FORMAT, command="bob measure roc"
    ),
    far_name="FPR",
)
def roc(ctx, scores, evaluation, **kwargs):
    process = figure.Roc(ctx, scores, evaluation, load.split)
    process.run()


@common_options.det_command(
    common_options.DET_HELP.format(
        score_format=SCORE_FORMAT, command="bob measure det"
    ),
    far_name="FPR",
)
def det(ctx, scores, evaluation, **kwargs):
    process = figure.Det(ctx, scores, evaluation, load.split)
    process.run()


@common_options.epc_command(
    common_options.EPC_HELP.format(
        score_format=SCORE_FORMAT, command="bob measure epc"
    )
)
def epc(ctx, scores, **kwargs):
    process = figure.Epc(ctx, scores, True, load.split)
    process.run()


@common_options.hist_command(
    common_options.HIST_HELP.format(
        score_format=SCORE_FORMAT, command="bob measure hist"
    ),
    far_name="FPR",
)
def hist(ctx, scores, evaluation, **kwargs):
    process = figure.Hist(ctx, scores, evaluation, load.split)
    process.run()


@common_options.evaluate_command(
    common_options.EVALUATE_HELP.format(
        score_format=SCORE_FORMAT, command="bob measure evaluate"
    ),
    criteria=CRITERIA,
    far_name="FPR",
)
def evaluate(ctx, scores, evaluation, **kwargs):
    common_options.evaluate_flow(
        ctx, scores, evaluation, metrics, roc, det, epc, hist, **kwargs
    )


@common_options.multi_metrics_command(
    common_options.MULTI_METRICS_HELP.format(
        names="FtA, FAR, FRR, FMR, FMNR, HTER",
        criteria=CRITERIA,
        score_format=SCORE_FORMAT,
        command="bob measure multi-metrics",
    ),
    criteria=CRITERIA,
    far_name="FPR",
)
def multi_metrics(ctx, scores, evaluation, protocols_number, **kwargs):
    ctx.meta["min_arg"] = protocols_number * (2 if evaluation else 1)
    process = figure.MultiMetrics(ctx, scores, evaluation, load.split)
    process.run()
