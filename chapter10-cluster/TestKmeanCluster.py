# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/2/22 12:37
# Description:
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib import cm
from sklearn.metrics import silhouette_samples

# 生成150个样本，并且每个样本与两个特征相关，总共三类， 每一类样本之间的方差为0.5
X, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)


def plot_data():
    plt.scatter(X[:, 0], X[:, 1], c='white', marker='o', edgecolors='black', s=50)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid()
    plt.tight_layout()
    plt.show()


# 使用k均值算法处理上述生成的数据
# 设置簇为3，n_init=10 独立10次运行j均值算法 每次运行最大迭代次数为300 tol用于控制簇内误差平方和的变化
# 这种算法啊的缺点就是 事先不确定数据到底有多少类
km = KMeans(n_clusters=3, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
y_km = km.fit_predict(X)
class_labels = np.unique(y_km)
n_clusters = class_labels.shape[0]
# 计算每个簇的轮廓系数
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(class_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    # 取出对应的颜色
    color = cm.jet(float(i) / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, edgecolor='none', color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)

silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color='red', linestyle='--')
plt.yticks(yticks, class_labels + 1)
plt.ylabel("Cluster")
plt.xlabel("Silhouette coefficient")
plt.tight_layout()
plt.show()


def plot_kmean_cluster():
    colors = ['lightgreen', 'orange', 'lightblue']
    markers = ['s', 'o', 'v']
    for i in range(3):
        plt.scatter(X[y_km == i, 0], X[y_km == i, 1], s=50, c=colors[i], marker=markers[i], edgecolors='black',
                    label=f'Cluster {i + 1}')
    # 标记簇类中心点
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=250, c='red', marker='*', edgecolors='black',
                label="Centroids")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.tight_layout()
    plt.show()


plot_kmean_cluster()

distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)

# 肘方法 就是识别失真变化最迅速的拐点 就是最佳簇点，就是对应的类别数
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel("Number of clusters")
plt.ylabel("Distortion")
plt.tight_layout()
plt.show()
