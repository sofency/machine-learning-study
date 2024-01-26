# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/1/26 10:36
# Description: PCA进行主成分分析
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from chapter03.common.function import plot_decision_regions


# 协方差分析主成分图
def plot_cov(eigen_vals):
    tot = sum(eigen_vals)
    var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)

    plt.bar(range(1, 14), var_exp, align='center', label="Individual explained variance")
    plt.step(range(1, 14), cum_var_exp, where='mid', label='Cumulative explained variance')
    plt.ylabel("Explained variance ratio")
    plt.xlabel("principal component index")

    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def plot_two_feature(X_train_pca, y_train):
    colors = ['r', 'b', 'g']
    markers = ['o', 's', '^']
    for label, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(X_train_pca[y_train == label, 0], X_train_pca[y_train == label, 1], c=c, label=f'Class {label}',
                    marker=m)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()


# 自定义的pca主成分分析
def custom_pca(df_wine, X_train_std, y_train):
    # 计算协方差矩阵
    cov_mat = np.cov(X_train_std.T)
    # 得到协方差矩阵的特征值和特征向量
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    # 特征值和特征向量对
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]

    eigen_pairs.sort(key=lambda k: k[0], reverse=True)
    # 这里只选取两个主成分
    w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
    # 二维散点图
    X_train_pca = X_train_std.dot(w)

    plot_two_feature(X_train_pca, y_train)

    loadings = eigen_vecs * np.sqrt(eigen_vals)
    # 评估主成分载荷
    evaluate_feature_contribution(df_wine, loadings)


def evaluate_feature_contribution(df_wine, loadings):
    fig, ax = plt.subplots()
    ax.bar(range(13), loadings[:, 0], align='center')
    ax.set_ylabel("Loading for PC 1")
    ax.set_xticks(range(13))
    ax.set_xticklabels(df_wine.columns[1:], rotation=90)
    plt.ylim([-1, 1])
    plt.tight_layout()
    plt.show()


def sklearn_pca(df_wine, X_train_std, y_train):
    pca = PCA(n_components=2)
    lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
    X_train_pca = pca.fit_transform(X_train_std)
    # X_test_pca = pca.transform(X_test_std)
    lr.fit(X_train_pca, y_train)

    plot_decision_regions(X_train_pca, y_train, classifier=lr)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()

    sklearn_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    evaluate_feature_contribution(df_wine, sklearn_loadings)


# 协方差的计算
def test_cov():
    array = np.array([
        [1, 2, 3, 4, 5],
        [6, 6, 8, 6, 9],
        [2, 4, 1, 2, 1]
    ])
    median = np.mean(array, axis=0)
    array_sub = array - median[:, np.newaxis].T
    # 2 是 3 - 1
    result = np.matmul(array_sub.T, array_sub) / 2
    print(result)
    print("=====精简实现======")
    print(np.cov(array.T))


if __name__ == '__main__':
    df_wine = pd.read_csv("../data/wine.data")
    df_wine.columns = ["Class label", "Alcohol", "Malic acid", "Ash",
                       "Alcalinity of ash", "Magnesium",
                       "Total phenols", "Flavanoids",
                       "Nonflavanoid phenols",
                       "Proanthocyanins", "Color intensity", "Hue",
                       "OD280/OD315 of diluted wines", "Proline"]
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)
    sklearn_pca(df_wine, X_train_std, y_train)
