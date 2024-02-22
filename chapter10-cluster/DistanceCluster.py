# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/2/22 17:51
# Description:
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

np.random.seed(123)
variables = ["X", "Y", "Z"]
labels = ["ID_0", "ID_1", "ID_2", "ID_3", "ID_4"]
# np.random.random_sample([5,3]) 随机生成[5,3]的矩阵 只为0-1之间
X = np.random.random_sample([5, 3]) * 10

df = pd.DataFrame(X, columns=variables, index=labels)
# pdist(df, metric='euclidean')计算的是压缩距离矩阵  squareform(pdist(df, metric='euclidean')) 还原为对称距离矩阵
row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')), columns=labels, index=labels)
print(row_dist)

# linkage 原理: 函数的输入是一个距离矩阵，其中每个元素表示数据点之间的相似性或距离，
# 此外还需要指定计算簇之间距离的方法 ("single", "complete", "average", "weighted", "centroid"),
# 迭代过程：在每次迭代中，算法会选择最近的两个簇（根据指定的距离计算方法）并将它们合并为一个新的簇。（并且会给簇在原有数据点标签上以此累加）
#          然后，这个新簇将与其他簇的距离重新计算。这个过程一直持续到只剩下一个簇为止
# 输出：该函数返回一个 (n-1) * 4 的矩阵，其中 n 是原始数据点的数量。这个矩阵包含了每次迭代中合并的簇的信息，
#       包括合并前簇的标签、合并后簇的标签、合并前簇之间的距离以及合并时涉及的数据点的数量。
row_clusters = linkage(pdist(df, metric='euclidean'), method='complete')

# 显示连接矩阵
row_dendr = dendrogram(row_clusters, labels=labels)
plt.tight_layout()
plt.ylabel("Euclidean distance")
plt.show()

# 通过Scikit-learn 进行凝聚聚类
ac = AgglomerativeClustering(n_clusters=3, linkage='complete')
labels = ac.fit_predict(X)
print(f"Cluster labels: {labels}")

