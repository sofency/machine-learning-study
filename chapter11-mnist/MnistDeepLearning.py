# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/2/23 08:17
# Description:
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

# fetch_openml 函数从openml网站下载数据集
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X.values
y = y.values

# 像素归一化 (像素范围-1 ~ 1 之间)
X = ((X / 255.) - .5) * 2
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()

for i in range(10):
    img = X[y == str(i)][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

