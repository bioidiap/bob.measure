import numpy
numpy.random.seed(42)
import bob.measure
from matplotlib import pyplot

cmc_scores = []
for probe in range(1000):
  positives = numpy.random.normal(1, 1, 1)
  negatives = numpy.random.normal(0, 1, 19)
  cmc_scores.append((negatives, positives))
for probe in range(1000):
  negatives = numpy.random.normal(-1, 1, 10)
  cmc_scores.append((negatives, None))

bob.measure.plot.detection_identification_curve(cmc_scores, rank=1, logx=True)
pyplot.xlabel('False Alarm Rate')
pyplot.xlim([0.0001, 1])
pyplot.ylabel('Detection & Identification Rate (%)')
pyplot.ylim([0,1])