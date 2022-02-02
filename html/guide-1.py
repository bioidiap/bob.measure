import numpy
numpy.random.seed(42)
import bob.measure
from matplotlib import pyplot

positives = numpy.random.normal(1,1,100)
negatives = numpy.random.normal(-1,1,100)
npoints = 100
bob.measure.plot.roc(negatives, positives, npoints, color=(0,0,0), linestyle='-', label='test')
pyplot.grid(True)
pyplot.xlabel('FPR (%)')
pyplot.ylabel('FNR (%)')
pyplot.title('ROC')