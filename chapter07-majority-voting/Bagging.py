# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/2/19 14:13
# Description:
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df_wine = pd.read_csv("../data/wine.data")

df_wine.columns = ["Class label", "Alcohol", "Malic acid", "Ash",
                   "Alcalinity of ash", "Magnesium",
                   "Total phenols", "Flavanoids",
                   "Nonflavanoid phenols",
                   "Proanthocyanins", "Color intensity", "Hue",
                   "OD280/OD315 of diluted wines", "Proline"]

# 删除掉类别为1的样本 并且只选取两个特征
df_wine = df_wine[df_wine["Class label"] != 1]
y = df_wine["Class label"].values
X = df_wine[['Alcohol', 'OD280/OD315 of diluted wines']].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

tree = DecisionTreeClassifier(criterion='entropy', random_state=1, max_depth=None)

# estimator: 指定个体评估器，即基础学习器
# n_estimators: 指定bagging算法中基础模型的数量，即要集成的个体评估器的数量, 增加基础模型的数量可能会提高模型的泛化能力，但也会增加计算成本
# max_samples: 指定每次抽样时抽取的样本量
# max_features: 默认为1.0，表示使用全部特征 这个参数可以控制每个基础模型所使用的特征子集大小。
# bootstrap: 表示样本是否为有放回抽样
# bootstrap_features: 表示特征是否为有放回抽样
bag = BaggingClassifier(estimator=tree, n_estimators=500, max_samples=1.0, max_features=1.0, bootstrap=True,
                        bootstrap_features=False, n_jobs=1, random_state=1)

tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)

print(f"Decision tree train/test accuracies {tree_train:.3f}/{tree_test:.3f}")

bag = bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)

bag_train = accuracy_score(y_train, y_train_pred)
bag_test = accuracy_score(y_test, y_test_pred)

print(f"Bagging train/test accuracies {bag_train:.3f}/{bag_test:.3f}")

# 绘制决策边界
x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1
y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=1, ncols=2, sharex='col', sharey='row', figsize=(8, 3))

for idx, clf, tt in zip([0, 1], [tree, bag], ['Decision tree', 'Bagging']):
    clf.fit(X_train, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx].scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='blue', marker='^')
    axarr[idx].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='green', marker='o')
    axarr[idx].set_title(tt)

axarr[0].set_ylabel('OD280/OD315 of diluted wines', fontsize=12)
plt.tight_layout()
plt.text(0,-0.2, s='Alcohol', ha='center', va='center', fontsize=12, transform=axarr[1].transAxes)
plt.show()
