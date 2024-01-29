# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/1/29 16:45
# Description:
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from common.load_data import load_data
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
# 还在实验中 不推荐使用 后续的api可能改变
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
# 混淆矩阵
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer, f1_score

pipeline = make_pipeline(StandardScaler(), SVC(random_state=1))
X_train, X_test, y_train, y_test = load_data()

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
# param_range = scipy.stats.loguniform(0.0001, 1000.0)

param_grid = [
    {"svc__C": param_range, "svc__kernel": ["linear"]},
    {"svc__C": param_range, "svc__gamma": param_range, "svc__kernel": ["rbf"]}
]
# 自制评分准则 下面的scoring 可以指定为如下 scoring=scorer
scorer = make_scorer(f1_score, pos_label=0)
# cv 是使用多少折交叉验证
models = {
    "random": RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid, scoring='accuracy', refit=True,
                                 n_iter=20, cv=10, random_state=1, n_jobs=-1),
    "halving": HalvingRandomSearchCV(pipeline, param_distributions=param_grid, n_candidates='exhaust',
                                     resource='n_samples', factor=1.5, random_state=1, n_jobs=-1),
    "grid": GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='accuracy', cv=10, refit=True, n_jobs=-1)
}

rs = models.get("halving")
rs = rs.fit(X_train, y_train)
# 获取到最好的模型参数进行预测
y_pred = rs.best_estimator_.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)


def plot_confmat(confmat):
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.5)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va="center", ha="center")
    ax.xaxis.set_ticks_position("bottom")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()


plot_confmat(confmat)

print(rs.best_score_)
print(rs.best_params_)
