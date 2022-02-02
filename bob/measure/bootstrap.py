import numpy
# had to install sklearn
from sklearn.utils import resample
# from matplotlib import pyplot

# create dataset
def create_groundtruth(positives, negatives, nb_samples):
    data = numpy.zeros(nb_samples)
    data[0:positives] = 1
    numpy.random.shuffle(data)
    return data

def create_model_output(TP, TN, FP, FN, groundtruth):
    model_output = numpy.ndarray.copy(groundtruth)
    where_0 = numpy.where(groundtruth == 0)
    where_1 = numpy.where(groundtruth == 1)
    model_output[where_0[0][:FP]] = 1
    model_output[where_1[0][:FN]] = 0
    return model_output

def bootstrap(nb_iterations, groundtruth_model_output, performance_measure):
    # # configure bootstrap
    nb_samples_boot = int(groundtruth_model_output.shape[0] * 0.80)
    stats = numpy.zeros(nb_iterations)
    for i in range(nb_iterations):
        boot_data = resample(data_result, n_samples = nb_samples_boot)
        TPboot = numpy.where(boot_data == [1, 1])[0].shape[0]
        TNboot = numpy.where(boot_data == [0, 0])[0].shape[0]
        FPboot = numpy.where(boot_data == [0, 1])[0].shape[0]
        FNboot = numpy.where(boot_data == [1, 0])[0].shape[0]
        if performance_measure == 'precision' :
            stats[i] = TPboot / (TPboot + FPboot)
        elif performance_measure == 'recall' :
            stats[i] = TPboot / (TPboot + FNboot)
        elif performance_measure == 'specificity' :
            stats[i] = TNboot / (TNboot + FPboot)
        elif performance_measure == 'accuracy' :
            stats[i] = (TPboot + TNboot) / (TPboot + TNboot + FPboot + FNboot)
        elif performance_measure == 'jaccard' :
            stats[i] = (TPboot) / (TPboot + FPboot + FNboot)
        elif performance_measure == 'F1-score' :
            stats[i] = (2 * TPboot) / (2 * TPboot + FPboot + FNboot)
    return stats
    
TP = 8
TN = 72 
FP = 18
FN = 2
positives = TP + FN
negatives = TN + FP
gt = create_groundtruth(positives, negatives, positives + negatives)
print(gt)
mo = create_model_output(TP, TN, FP, FN, gt)
print(mo)
data_result = numpy.stack((gt, mo), axis = 1)
print(data_result.shape[0])
print(numpy.sort(bootstrap(1000, data_result, 'precision')))

