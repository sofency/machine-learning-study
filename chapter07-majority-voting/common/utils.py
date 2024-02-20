# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/1/31 12:59
# Description:

from scipy.special import comb
import math
import numpy as np
import matplotlib.pyplot as plt


# 集成分类器的错误率的概率质量函数
def ensemble_error(n_classifier, error):
    k_start = int(math.ceil(n_classifier / 2.))
    probs = [comb(n_classifier, k) * error ** k * (1 - error) ** (n_classifier - k) for k in
             range(k_start, n_classifier + 1)]
    return sum(probs)


def plot_error_ratio():
    error_range = np.arange(0.0, 1.01, 0.01)
    ens_errors = [ensemble_error(n_classifier=11, error=error) for error in error_range]
    plt.plot(error_range, ens_errors, label="Ensemble error", linewidth=2)
    plt.plot(error_range, error_range, linestyle='--', label="Base error", linewidth=2)
    plt.xlabel("Base error")
    plt.ylabel("Base/Ensemble error")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.5)
    plt.show()


plot_error_ratio()
# [0.4 0.6] 先统计种类的个数，然后 乘以对应的概率
print(np.bincount([0, 0, 1], weights=[0.2, 0.2, 0.6]))

