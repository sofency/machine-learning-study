# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/2/19 14:10
# Description:
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV
import common.MajorityVoteClassifier as MajorityVoteClassifier

iris = datasets.load_iris()
# 只选择萼片宽度和花瓣长度两个特征
X, y = iris.data[50:, [1, 2]], iris.target[50:]
le = LabelEncoder()
# 转换字符串类别为数字
y = le.fit_transform(y)
# stratify 如果指定了这个参数，那么训练集和测试集中的类别比例将与原始数据集中的类别比例相同。这对于确保训练集和测试集在类别分布上的一致性非常有用。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, stratify=y)

# 减少C, 可以增加正则化强度，可以减少过拟合
clf1 = LogisticRegression(penalty='l2', C=0.001, solver='lbfgs', random_state=1)
# 决策树深度越深，决策边界越复杂，越容易过拟合
clf2 = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)
# 欧式距离和曼哈顿距离的推广 见树81页 p = 2 就是欧式距离
clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')

pipe1 = Pipeline([['sc', StandardScaler()], ['clf', clf1]])
pipe3 = Pipeline([['sc', StandardScaler()], ['clf', clf3]])

clf_labels = ['Logistic regression', 'Decision tree', 'KNN']
print("10-fold cross validation:\n")
for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
    # print(clf, label)
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='roc_auc')
    print(f"ROC AUC: {scores.mean():.2f} (+/- {scores.std():.2f}) [{label}]")

# 加入多票选举后
print("after adding majority voting\n")

mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])
clf_labels += ["Majority voting"]
all_clf = [pipe1, clf2, pipe3, mv_clf]
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='roc_auc')
    print(f"ROC AUC: {scores.mean():.2f} (+/- {scores.std():.2f}) [{label}]")

colors = ['black', 'orange', 'blue', 'green']
line_styles = [':', '--', '-.', '-']
# tpr = tp / (tp+fn)  fpr = tp / (fp + tn)
for clf, label, clr, ls in zip(all_clf, clf_labels, colors, line_styles):
    y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr, color=clr, linestyle=ls, label=f'{label} (auc = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid(alpha=0.5)
plt.xlabel('False positive rate (FPR)')
plt.ylabel("True positive rate (TPR)")
plt.show()

# 画出决策边界
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
from itertools import product

x_min = X_train_std[:, 0].min() - 1
x_max = X_train_std[:, 0].max() + 1
y_min = X_train_std[:, 1].min() - 1
y_max = X_train_std[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
# nrows=2 ncols=2 表示图有2 * 2 = 4个 我们可以通过axarr进行设置每个图的参数信息
f, axarr = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row', figsize=(7, 5))
# product([0,1],[0,1]) 生成坐标位置 [0,0] [0,1] [1,0], [1,1]
for idx, clf, tt in zip(product([0, 1], [0, 1]), all_clf, clf_labels):
    clf.fit(X_train_std, y_train)
    # xx.ravel() 将多维数组展平为一维数组
    # a = np.array([1, 2, 3])
    # b = np.array([4, 5, 6])
    # # 使用 np.c_ 按列连接这两个数组
    # c = np.c_[a, b]
    # c = [[1 4]
    #      [2 5]
    #      [3 6]]
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # contourf 绘制填充等高线图 边界线
    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3)

    axarr[idx[0], idx[1]].scatter(X_train_std[y_train == 0, 0], X_train_std[y_train == 0, 1],
                                  c='blue', marker='^', s=50)

    axarr[idx[0], idx[1]].scatter(X_train_std[y_train == 1, 0], X_train_std[y_train == 1, 1],
                                  c='green', marker='o', s=50)

    axarr[idx[0], idx[1]].set_title(tt)

plt.text(-3.5, -5., s="Sepal width [standardized]", ha='center', va='center', fontsize=12)
plt.text(-12.5, 4.5, s="Petal length [standardized]", ha='center', va='center', fontsize=12, rotation=90)
plt.show()

# mv_clf.get_params() 查看模型中的参数
# 网格搜索观察为啥上述C 要设置为0.001
params = {'decisiontreeclassifier__max_depth': [1, 2], 'pipeline-1__clf__C': [0.001, 0.1, 100.0]}
grid = GridSearchCV(estimator=mv_clf, param_grid=params, cv=10, scoring='roc_auc')
grid.fit(X_train, y_train)

for r, _ in enumerate(grid.cv_results_['mean_test_score']):
    mean_score = grid.cv_results_['mean_test_score'][r]
    std_dev = grid.cv_results_['std_test_score'][r]
    params = grid.cv_results_['params'][r]
    print(f"{mean_score:.3f} +/- {std_dev:.2f}  {params}")

print(f"Best Parameters : {grid.best_params_}")
print(f"Roc AUC : {grid.best_score_:.2f}")
