import numpy
from matplotlib import pyplot as plt
import bob.measure.plot

# We simulate scores for 2 hypothethical systems.
# Scores are beta distributed for a "perfect" example.

nb_samples1 = 200  # Sample size, higher makes it more precise (thinner CI)
a1 = 6
b1 = 10

nb_samples2 = 100  # Sample size, higher makes it more precise (thinner CI)
a2 = 7
b2 = 10
numpy.random.seed(42)

negatives1 = numpy.random.beta(a=a1, b=b1, size=nb_samples1)
# ph = plt.hist(negatives, bins=100, alpha=0.3, label="Negatives")
# plt.hist(negatives, bins=100, color=h1[2][0].get_facecolor()[:3], histtype="step")

positives1 = numpy.random.beta(a=b1, b=a1, size=nb_samples1)
# nh = plt.hist(positives, bins=100, alpha=0.3, label="Positives")
# plt.hist(positives, bins=100, color=h1[2][0].get_facecolor()[:3], histtype="step")

negatives2 = numpy.random.beta(a=a2, b=b2, size=nb_samples2)
# ph = plt.hist(negatives, bins=100, alpha=0.3, label="Negatives")
# plt.hist(negatives, bins=100, color=h1[2][0].get_facecolor()[:3], histtype="step")

positives2 = numpy.random.beta(a=b2, b=a2, size=nb_samples2)
# nh = plt.hist(positives, bins=100, alpha=0.3, label="Positives")
# plt.hist(positives, bins=100, color=h1[2][0].get_facecolor()[:3], histtype="step")

# plt.title("Scores (i.i.d. samples)")

# We now compute the ROC curve with the confidence interval
axes = ("tpr", "fpr")
AXES = [k.upper() for k in axes]
with bob.measure.plot.tight_roc_layout(axes=AXES):
    obj1, auc1 = bob.measure.plot.roc_ci(
        negatives1,
        positives1,
        axes=axes,
    )
    obj2, auc2 = bob.measure.plot.roc_ci(
        negatives2,
        positives2,
        axes=axes,
    )
    plt.legend(
        (obj1, obj2),
        (
            f"System 1 (AUC: {auc1[0]:.2f}"
            f" - 95% CI: {auc1[1]:.2f}-{auc1[2]:.2f})",
            f"System 2 (AUC: {auc2[0]:.2f}"
            f" - 95% CI: {auc2[1]:.2f}-{auc2[2]:.2f})",
        ),
        loc="best",
        fancybox=True,
        framealpha=0.7,
    )

plt.show()
