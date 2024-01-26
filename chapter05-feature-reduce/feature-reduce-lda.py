# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/1/26 14:41
# Description:
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from chapter03.common.function import plot_decision_regions


# 协方差分析主成分图
def plot_cov(eigen_vals):
    tot = sum(eigen_vals.real)
    var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)

    plt.bar(range(1, 14), var_exp, align='center', label="Individual explained variance")
    plt.step(range(1, 14), cum_var_exp, where='mid', label='Cumulative explained variance')
    plt.ylabel("Explained variance ratio")
    plt.xlabel("principal component index")

    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def plot_three_feature(X_train_pca, y_train):
    colors = ['r', 'b', 'g']
    for label, c in zip(np.unique(y_train), colors):
        # Plot for PC1
        plt.scatter(X_train_pca[y_train == label, 0], X_train_pca[y_train == label, 2], c=c, label=f'Class {label}',
                    marker='o')
    plt.xlabel("PC 1 and PC 2")
    plt.ylabel("PC 3")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()


def custom_lda(X_train_std, y_train):
    # 设置精度为4
    np.set_printoptions(precision=4)
    mean_vecs = []
    for label in range(1, 4):
        # 找出每个特征标签为label的所有列均值  (即每个特征的均值)
        mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
        print(f'MV {label}: {mean_vecs[label - 1]}\n')

    d = 13
    # 类内散布矩阵
    S_W = np.zeros((d, d))
    for label, mv in zip(range(1, 4), mean_vecs):
        class_scatter = np.zeros((d, d))
        for row in X_train_std[y_train == label]:
            row, mv = row.reshape(d, 1), mv.reshape(d, 1)
            class_scatter += (row - mv).dot((row - mv).T)

        S_W += class_scatter
    print("Within-class scatter matrix: ", f'{S_W.shape[0]}x{S_W.shape[1]}')

    # 类间散布矩阵
    mean_overall = np.mean(X_train_std, axis=0)
    mean_overall = mean_overall.reshape(d, 1)
    S_B = np.zeros((d, d))
    for i, mean_vec in enumerate(mean_vecs):
        n = X_train_std[y_train == i + 1, :].shape[0]
        mean_overall = mean_vec.reshape(d, 1)
        S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
    print("Between-class scatter matrix: ", f'{S_B.shape[0]}x{S_B.shape[1]}')

    # LDA的矩阵分解是针对(Sw)^(-1)*Sb
    eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

    # 特征值降序
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
    eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
    print("Eigenvalues in descending order:\n")
    for eigen_val in eigen_pairs:
        print(eigen_val[0])

    plot_cov(eigen_vals)

    w = np.hstack(
        (eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis], eigen_pairs[2][1][:, np.newaxis]))
    X_train_lda = X_train_std.dot(w)

    plot_three_feature(X_train_lda, y_train)


def sklearn_lda(X_train_std, y_train):
    # n_components cannot be larger than min(n_features, n_classes - 1)
    lda = LDA(n_components=2)
    print(X_train_std.shape)
    X_train_lda = lda.fit_transform(X_train_std, y_train)
    lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
    lr.fit(X_train_lda, y_train)
    print(lr.classes_)
    plot_decision_regions(X_train_lda, y_train, classifier=lr)
    plt.xlabel("LD 1")
    plt.ylabel("LD 2")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()


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
    sklearn_lda(X_train_std, y_train)
