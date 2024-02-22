# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/2/22 18:46
# Description:
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.patheffects as PathEffects

digits = load_digits()
fig, ax = plt.subplots(nrows=1, ncols=4)
for i in range(4):
    ax[i].imshow(digits.images[i], cmap='Greys')
plt.show()

X_digits = digits.data
y_digits = digits.target
print("原来的形状", X_digits.shape)
# 降维处理
tsne = TSNE(n_components=2, init='pca', random_state=123)
X_digits_tsne = tsne.fit_transform(X_digits)


def plot_projection(x, colors):
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    for i in range(10):
        plt.scatter(x[colors == i, 0], x[colors == i, 1])
    # 写出对应簇的target
    for i in range(10):
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground='w'),
            PathEffects.Normal()
        ])


plot_projection(X_digits_tsne, y_digits)
plt.show()
