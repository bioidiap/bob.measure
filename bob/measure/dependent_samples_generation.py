#!/usr/bin/env python
# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


def generate_groundtruth(first_sample_pb, keep_state_pb, nb_samples):
    gt = np.zeros(nb_samples)
    prev_value = np.random.choice([1, 0], p=[first_sample_pb, 1 - first_sample_pb])
    gt[0] = prev_value
    for i in range(1, nb_samples):
        new_value = np.random.choice([prev_value, 1 - prev_value], p=[keep_state_pb, 1 - keep_state_pb])
        gt[i] = new_value
        prev_value = new_value
    return gt

def generate_ml(accuracy, groundtruth, nb_systems):
    ml = np.zeros([nb_systems, groundtruth.size])
    for system in range(nb_systems):
        print(system)
        for i in range(groundtruth.size):
            ml[system][i] = np.random.choice([groundtruth[i] ,1 - groundtruth[i]], p=[accuracy, 1 - accuracy])
    return ml

def compute_accuracy(ml, gt):
    return np.mean(ml == gt, axis=1)



gt = generate_groundtruth(1, 1, 10000)
ml = generate_ml(0.5, gt, 10000)
acc = compute_accuracy(ml, gt)


fig, ax = plt.subplots(nrows=2, ncols=2, tight_layout=True)
n_bins = 10
ax[0,0].hist(acc, bins=n_bins)
n_bins = 20
ax[0,1].hist(acc, bins=n_bins)
n_bins = 50
ax[1,0].hist(acc, bins=n_bins)
n_bins = 100
ax[1,1].hist(acc, bins=n_bins)
plt.show()