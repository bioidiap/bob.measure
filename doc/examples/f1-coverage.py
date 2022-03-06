#!/usr/bin/env python
# coding=utf-8

"""F1-score coverage for a variety of Precision vs. Recall values

In this Monte-Carlo simulation, we evaluate the coverage of the posterior
F1-score distribution considering both Precision and Recall are issued from
true binomial distributions.

We note that not all values of precision vs. recall are possible, because of
their definition:

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)

For the purpose of this exercise, we assume however this is not the case and
any combination of precision and recall values are possible.  This assumption
also simplifies the calculations.
"""

import itertools
import numpy
import matplotlib.pyplot as plt

from bob.measure.credible_region import f1_score

# number of samples on each experiment
positives = 100
negatives = 100

mc_simulation = 10000  # size of the simulated samples
coverage = 0.95

# PROGRAM STARTS HERE

# the probabilities "p" for the binomial sources
# we only "scan" through possible probabilities given the size of the
# population
precision_ps = numpy.arange(
    2 / (positives + negatives), 1, 2 / (positives + negatives)
)
recall_ps = numpy.arange(1 / positives, 1, 1 / positives)

# we iterate over all possibilities.  each of these is a point on the final
# coverage plot
coverage_matrix = numpy.zeros((len(precision_ps), len(recall_ps)), dtype=float)

for (precision_idx, precision_p), (recall_idx, recall_p) in itertools.product(
    enumerate(precision_ps), enumerate(recall_ps)
):

    # the binomial **source** distribution for recall is the easiest, as that
    # depends only on the total number of positives available
    recall_binomial = numpy.random.binomial(
        n=positives, p=recall_p, size=mc_simulation
    )
    # notice that the actual simulation of recall should be that divided by
    # "positives", as that would be the "perceived" rate would the number of tp
    # over (tp+fn) be observed
    recall_binomial = recall_binomial / positives

    # the binomial **source** distribution for precision is trickier as we need
    # to evaluate (tp+fp), which is on the denominator of the equation.  we
    # calculate it from:
    # p = tp / (tp + fp)
    # p*tp + p*fp = tp
    # p*fp = tp - p*tp
    # fp = tp (1-p) / p
    tp = int(round(recall_p * positives))
    fp = int(round(tp * (1 - precision_p) / precision_p))

    # however, we are bound by the total number of negatives available in the
    # system.  we therefore check and only proceed if the system is "feasible"
    # within these bounds.  we assign a NaN to the coverage matrix we are
    # building in the case of unfeasibility.  later, we handle these missing
    # data points.
    if fp > negatives:
        # print(f"For P={precision_p:.2f}, FP={fp} > {negatives} (unfeasible)")
        coverage_matrix[precision_idx, recall_idx] = numpy.nan
        continue

    # otherwise, this system is feasible within our sample size constraints.
    # in this case, we just proceed with the simulation
    precision_binomial = numpy.random.binomial(
        n=(tp + fp), p=precision_p, size=mc_simulation
    )
    # notice that the actual simulation of precision should be that divided by
    # "tp+fp", as that would be the "perceived" rate would the number of tp
    # over (tp+fp) be observed
    precision_binomial = precision_binomial / (tp + fp)

    # the F1 **source** distributions for each case - this is what we expect to
    # approaximate with our posterior estimates
    source_f1 = (
        2
        * precision_binomial
        * recall_binomial
        / (
            precision_binomial
            + recall_binomial
            + ((precision_binomial + recall_binomial) == 0)
        )
    )
    assert max(source_f1) <= 1.0
    assert min(source_f1) >= 0.0

    # we now evaluate the "true" bounds of the F1 source distribution for the
    # required credible interval, alongside the width of these bounds
    left_half = (1 - coverage) / 2  # size of excluded (half) area

    # now we calculate the posterior F1 as estipulated by Goutte, and we
    # compare these bounds.
    fn = positives - tp
    mean, mode, post_lower, post_upper = f1_score(
        tp, fp, fn, lambda_=1.0, coverage=coverage, nb_samples=mc_simulation
    )
    outside = numpy.count_nonzero(source_f1 < post_lower) + numpy.count_nonzero(
        source_f1 > post_upper
    )
    estimated_coverage_width = 1 - (outside / mc_simulation)
    coverage_matrix[precision_idx, recall_idx] = estimated_coverage_width


# plots the surface
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

fig = plt.figure(figsize=plt.figaspect(0.5))
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax2 = fig.add_subplot(1, 2, 2)

X, Y = numpy.meshgrid(precision_ps, recall_ps)

surf = ax1.plot_surface(X, Y, coverage_matrix, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False, vmin=0, vmax=1)
ax1.set_xlabel("Precision")
ax1.set_ylabel("Recall")
ax1.set_zlabel("Coverage")
ax1.set_zlim(0, 1)
ax1.zaxis.set_major_locator(LinearLocator(10))
ax1.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

cont = ax2.contourf(X, Y, coverage_matrix, levels=100, cmap=cm.coolwarm, vmin=0, vmax=1)
ax2.set_xlabel("Precision")
ax2.set_ylabel("Recall")
ax2.set_title("Coverage")

# Add a color bar which maps values to colors.
fig.colorbar(cont)

plt.show()
