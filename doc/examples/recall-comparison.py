import numpy
import scipy.stats
from matplotlib import pyplot as plt
import bob.measure.credible_region

# system 1 performance
TP1 = 10
FN1 = 5
TN1 = 5
FP1 = 10

# system 2 performance
TP2 = 3
FN2 = 3
TN1 = 4
FP2 = 2

nb_samples = 10000  # Sample size, higher makes it more precise
lambda_ = 0.5  # use 1.0 for a flat prior, or 0.5 for Jeffrey's prior

# recall: TP / TP + FN, k=TP, l=FN

# now we calculate what is the probability that system 2's recall
# measurement is better than system 1
numpy.random.seed(42)  #for the monte-carlo simulation
prob = bob.measure.credible_region.compare_beta_posteriors(
    TP2, FN2, TP1, FN1, lambda_, nb_samples
)

# we then visualize the posteriors of the recall for both systems
# together with the probability that system 1 is better than system 2
x = numpy.linspace(0.01, 0.99, nb_samples)
pdf1 = scipy.stats.beta.pdf(x, TP1 + lambda_, FN1 + lambda_)
pdf2 = scipy.stats.beta.pdf(x, TP2 + lambda_, FN2 + lambda_)
plt.plot(x, pdf1, label=f"TP1 = {TP1}, FN1 = {FN1}")
plt.plot(x, pdf2, label=f"TP2 = {TP2}, FN2 = {FN2}")
plt.title(
    f"Posterior Recall - Monte Carlo - "
    f"($\mathbb{{P}}(R_2>R_1)$: {(100*prob):.0f}%)"
)
plt.grid()
plt.legend()
plt.show()