import numpy as np
import matplotlib.pyplot as plt
from credible_region import f1_score
import math
import scipy.stats
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def F1_percentage(nb_samples, nb_trials, coverage, lambda_, truepositivesprob, truenegativesprob):
    percentage_covered = np.empty([truepositivesprob.size, truenegativesprob.size])
    for index, tpprob in enumerate(truepositivesprob):
        tpbin = np.random.binomial(nb_trials, tpprob, nb_samples)
        fnbin = nb_trials - tpbin
        tp = int(round(tpprob*nb_trials))
        fn = nb_trials - tp
        for index1, tnprob in enumerate(truenegativesprob):
            tnbin = np.random.binomial(nb_trials, tnprob, nb_samples)
            fpbin = nb_trials - tnbin
            tn = int(round(tnprob*nb_trials))
            fp = nb_trials - tn
            F1scores = np.empty(nb_samples)
            for i in range(nb_samples) :
                if (tpbin[i]+fpbin[i]) == 0 :
                    precision = 0
                else :
                    precision = tpbin[i] / (tpbin[i]+fpbin[i])
                if (tpbin[i]+fnbin[i]) == 0 :
                    recall = 0
                else :
                    recall = tpbin[i] / (tpbin[i]+fnbin[i])
                if (precision + recall) == 0 :
                    F1 = 0
                else :
                    F1 = (2 * precision * recall)/(precision + recall)
                F1scores[i] = F1
            _,_, lowerCi, upperCi = f1_score(tp, fp, fn, lambda_, coverage, nb_samples)
            sorted_f1scores = np.sort(F1scores)
            percentage_covered[index][index1] = np.count_nonzero((upperCi > sorted_f1scores) & (sorted_f1scores > lowerCi)) / nb_samples
    return percentage_covered

truepositivesprob = np.arange(0.01, 1, 0.01) #x
truenegativesprob = np.arange(0.01, 1, 0.01) #y
nb_samples = 100
nb_trials = 100
coverage = 0.95
lambda_ = 0.5
percentage_covered = F1_percentage(nb_samples, nb_trials, coverage, lambda_, truepositivesprob, truenegativesprob) #z

# Plot the surface.
from matplotlib import cm
from matplotlib.ticker import LinearLocator

fig = plt.figure(figsize=plt.figaspect(0.5))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2)

X, Y = np.meshgrid(truepositivesprob, truenegativesprob)
surf = ax1.plot_surface(X, Y, percentage_covered, cmap=cm.coolwarm, linewidth=0)
cont = ax2.contourf(X, Y, percentage_covered, cmap=cm.coolwarm)

ax1.set_xlabel("TPR")
ax1.set_ylabel("TNR")
ax1.set_zlabel("Coverage")
ax2.set_xlabel("TPR")
ax2.set_ylabel("TNR")
ax2.set_title("Coverage")

# Customize the z axis.
ax1.set_zlim(0, 1)
ax1.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax1.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(cont, shrink=0.5, aspect=5)

plt.show()
