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
#         if (system % (nb_systems/10) == 0):
            # print progress each 10 percent
#             print((system / nb_systems) * 100, 'percent')
        for i in range(groundtruth.size):
            ml[system][i] = np.random.choice([groundtruth[i] ,1 - groundtruth[i]], p=[accuracy, 1 - accuracy])
    return ml

def compute_accuracy_k(ml, gt):
    return (ml == gt).sum(axis = 1)


for dependency_prob in np.linspace(0.1, 1, 9, endpoint=False):
#     print("===== Dependency probability is ", "{:.1f}".format(dependency_prob), "=====")
    gt = generate_groundtruth(0.5, dependency_prob, 1000)
    for accuracy in np.linspace(0.1, 1, 9, endpoint=False):
#         print("== Accuracy is ", "{:.1f}".format(accuracy), "==")
        ml = generate_ml(accuracy, gt, 10000)
        acc = compute_accuracy_k(ml, gt)
        fig, ax = plt.subplots(nrows=2, ncols=2, tight_layout=True)
        fig.suptitle('Dependency probability : ' + "{:.1f}".format(dependency_prob) + 'Accuracy : ' + "{:.1f}".format(accuracy), fontsize=16)
        n_bins = 10
        ax[0,0].hist(acc, bins=n_bins)
        ax[0,0].set_title(str(n_bins) + ' bins')
        n_bins = 20
        ax[0,1].hist(acc, bins=n_bins)
        ax[0,1].set_title(str(n_bins) + ' bins')
        n_bins = 50
        ax[1,0].hist(acc, bins=n_bins)
        ax[1,0].set_title(str(n_bins) + ' bins')
        n_bins = 100
        ax[1,1].hist(acc, bins=n_bins)
        ax[1,1].set_title(str(n_bins) + ' bins')
        plt.savefig('/idiap/home/amorais/Desktop/Dependent_samples_figures/' + 'Depprob' + "{:.1f}".format(dependency_prob) + 'acc' + "{:.1f}".format(accuracy) + '.png')
        plt.close()
