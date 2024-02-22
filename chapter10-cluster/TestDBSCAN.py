# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/2/22 18:22
# Description: 通过DBSCAN定位高密度区域
# DBSCAN缺点：在训练样本数量保持不变的情况下，如果数据集的特征增加，则会发生维数灾难，
# 对于特征增加的情况，在进行聚类之前 先使用主成分分析和t-SNE分析，将数据维度压缩到二维空间
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# 生成半月型数据
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

plt.scatter(X[:, 0], X[:, 1])
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()
plt.show()

# 可以聚类任意形状的数据集
db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
y_db = db.fit_predict(X)
plt.scatter(X[y_db == 0, 0], X[y_db == 0, 1], c='lightblue', edgecolors='black', marker='o', s=40, label='Cluster 1')
plt.scatter(X[y_db == 1, 0], X[y_db == 1, 1], c='red', edgecolors='black', marker='s', s=40, label='Cluster 2')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.tight_layout()
plt.show()

