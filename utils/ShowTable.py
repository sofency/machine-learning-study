# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/2/27 17:53
# Description:
import matplotlib.pyplot as plt
import numpy
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def plot(x, y, x_label, xlim=None, legend=[], figsize=[12, 8]):
    plt.figure(figsize=figsize)
    if len(legend) > 1:
        for i, temp_y in enumerate(y):
            plt.plot(temp_y, label=legend[i])
        plt.legend()
    else:
        plt.plot(y)
    plt.xticks(x)
    plt.xlabel(x_label)
    plt.xlim(xlim)
    plt.show()


def load_data(features, labels, batch_size, is_shuffle=True):
    index = np.arange(0, len(labels))
    if is_shuffle:
        np.random.shuffle(index)
    combine_dl = DataLoader(TensorDataset(features[index], labels[index]), batch_size=batch_size)
    return combine_dl
