from credible_region import beta
import numpy as np
import matplotlib.pyplot as plt
import random

def folding_monte_carlo(TP1, FP1, TP2, FP2, nb_samples):
    lower_scores = np.empty(nb_samples)
    upper_scores = np.empty(nb_samples)
    numTrials1 = (TP1 + FP1)
    tp1prob = TP1 / numTrials1
    tp1bin = np.random.binomial(numTrials1, tp1prob, nb_samples)
    fp1bin = numTrials1 - tp1bin
    numTrials2 = (TP2 + FP2)
    tp2prob = TP2 / numTrials2
    tp2bin = np.random.binomial(numTrials2, tp2prob, nb_samples)
    fp2bin = numTrials2 - tp2bin
    for i in range(nb_samples):
        _,_, lowerCiset1, upperCiset1 = beta(tp1bin[i], fp1bin[i], 0.5, 0.95)
        _,_, lowerCiset2, upperCiset2 = beta(tp2bin[i], fp2bin[i], 0.5, 0.95)
        lowerCi_mean = (lowerCiset1 + lowerCiset2) / 2
        lower_scores[i] = lowerCi_mean
        upperCi_mean = (upperCiset1 + upperCiset2) / 2
        upper_scores[i] = upperCi_mean
    return np.mean(lower_scores), np.mean(upper_scores)

def cross_folding(TP, FP, nb_samples, nb_cross_folding):
    lower_cross_folding_scores = np.empty(nb_cross_folding)
    upper_cross_folding_scores = np.empty(nb_cross_folding)
    for i in range(nb_cross_folding):
        numTrials = TP + FP
        pool = np.append(np.zeros(FP), np.ones(TP))
        np.random.shuffle(pool)
        set1, set2 = np.array_split(pool, 2)
        tp1 = np.count_nonzero(set1)
        fp1 = set1.size - tp1
        tp2 = np.count_nonzero(set2)
        fp2 = set2.size - tp2
        lower_cross_folding_scores[i], upper_cross_folding_scores[i] = folding_monte_carlo(tp1, fp1, tp2, fp2, nb_samples)
    return np.mean(lower_cross_folding_scores), np.mean(upper_cross_folding_scores)

low, upp = cross_folding(100, 100, 100, 20)
print(low)
print(upp)