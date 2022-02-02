import numpy
numpy.random.seed(42)
import bob.measure
from matplotlib import pyplot

cmc_scores = []
for probe in range(10):
    positives = numpy.random.normal(1, 1, 1)
    negatives = numpy.random.normal(0, 1, 19)
    cmc_scores.append((negatives, positives))
bob.measure.plot.cmc(cmc_scores, logx=False)
pyplot.grid(True)
pyplot.title('CMC')
pyplot.xlabel('Rank')
pyplot.xticks([1,5,10,20])
pyplot.xlim([1,20])
pyplot.ylim([0,100])
pyplot.ylabel('Probability of Recognition (%)')