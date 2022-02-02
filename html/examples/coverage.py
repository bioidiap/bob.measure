import bob.measure


def betacred_flat_prior(k, l, cov):
    return bob.measure.credible_region.beta(k, l, 1.0, cov)[2:]


def betacred_jeffreys_prior(k, l, cov):
    return bob.measure.credible_region.beta(k, l, 0.5, cov)[2:]


samples = 100  # number of samples simulated
coverage = 0.95

flat = bob.measure.curves.estimated_ci_coverage(
    betacred_flat_prior, n=samples, expected_coverage=coverage
)
bj = bob.measure.curves.estimated_ci_coverage(
    betacred_jeffreys_prior, n=samples, expected_coverage=coverage
)
cp = bob.measure.curves.estimated_ci_coverage(
    bob.measure.confidence_interval.clopper_pearson,
    n=samples,
    expected_coverage=coverage,
)
ac = bob.measure.curves.estimated_ci_coverage(
    bob.measure.confidence_interval.agresti_coull,
    n=samples,
    expected_coverage=coverage,
)
wi = bob.measure.curves.estimated_ci_coverage(
    bob.measure.confidence_interval.wilson,
    n=samples,
    expected_coverage=coverage,
)

from matplotlib import pyplot as plt

plt.plot(wi[0], 100 * wi[1], color="black", label="CI: Wilson (1927)")
plt.plot(cp[0], 100 * cp[1], color="orange", label="CI: Clopper-Pearson (1934)")
plt.plot(ac[0], 100 * ac[1], color="purple", label="CI: Agresti-Coull (1998)")
plt.plot(
    flat[0], 100 * flat[1], color="blue", label="CR: Beta + Flat Prior (2005)"
)
plt.plot(
    bj[0], 100 * bj[1], color="green", label="CR: Beta + Jeffreys Prior (2005)"
)

# Styling
plt.ylabel(f"Coverage for {100*coverage:.0f}% CR/CI")
plt.xlabel(f"Success rate (p)")
plt.title(f"Estimated coverage n={samples}")
plt.ylim([75, 100])
plt.hlines(100 * coverage, bj[0][0], bj[0][-1], color="red", linestyle="dashed")
plt.grid()
plt.legend()
plt.show()
