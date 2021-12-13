from credible_region import beta
from credible_region import beta_posterior
import numpy as np
import matplotlib.pyplot as plt
import random

def cross_folding(TP1, FP1, TP2, FP2, nb_samples):
    # first cross folding
    numTrials1 = TP1 + FP1
    pool1 = np.append(np.zeros(FP1), np.ones(TP1))
    np.random.shuffle(pool1)
    set1_1, set1_2 = np.array_split(pool1, 2)
    tp1_1 = np.count_nonzero(set1_1)
    fp1_1 = set1_1.size - tp1_1
    tp1_2 = np.count_nonzero(set1_2)
    fp1_2 = set1_2.size - tp1_2
    beta_post1_1 = beta_posterior(tp1_1, fp1_1, 0.5, nb_samples)
    beta_post1_2 = beta_posterior(tp1_2, fp1_2, 0.5, nb_samples)
    #second cross folding
    numTrials2 = TP2 + FP2
    pool2 = np.append(np.zeros(FP2), np.ones(TP2))
    np.random.shuffle(pool2)
    set2_1, set2_2 = np.array_split(pool2, 2)
    tp2_1 = np.count_nonzero(set2_1)
    fp2_1 = set2_1.size - tp2_1
    tp2_2 = np.count_nonzero(set2_2)
    fp2_2 = set2_2.size - tp2_2
    beta_post2_1 = beta_posterior(tp2_1, fp2_1, 0.5, nb_samples)
    beta_post2_2 = beta_posterior(tp2_2, fp2_2, 0.5, nb_samples)
    return np.count_nonzero(beta_post1_1 > beta_post1_2) / nb_samples, np.count_nonzero(beta_post2_1 > beta_post2_2) / nb_samples

first_cross_comp, second_cross_comp = cross_folding(50, 50, 52, 48, 100)
print(first_cross_comp)
print(second_cross_comp)

# def cross_folding(TP1, FP1, TP2, FP2, nb_samples):
#     lower_scores = np.empty(nb_samples)
#     upper_scores = np.empty(nb_samples)
#     numTrials1 = (TP1 + FP1)
#     tp1prob = TP1 / numTrials1
#     tp1bin = np.random.binomial(numTrials1, tp1prob, nb_samples)
#     fp1bin = numTrials1 - tp1bin
#     numTrials2 = (TP2 + FP2)
#     tp2prob = TP2 / numTrials2
#     tp2bin = np.random.binomial(numTrials2, tp2prob, numSamples)
#     fp2bin = numTrials2 - tp2bin
#     for i in range(nb_samples):
#         _,_, lowerCiset1, upperCiset1 = beta(tp1bin[i], fp1bin[i], 0.5, 0.95)
#         _,_, lowerCiset2, upperCiset2 = beta(tp2bin[i], fp2bin[i], 0.5, 0.95)
#         lowerCi_mean = (lowerCiset1 + lowerCiset2) / 2
#         lower_scores[i] = lowerCi_mean
#         upperCi_mean = (upperCiset1 + upperCiset2) / 2
#         upper_scores[i] = upperCi_mean
#     return lower_scores, upper_scores

# low, upp = cross_folding(50, 50, 52, 48, 100)
# print(low)
# print(upp)