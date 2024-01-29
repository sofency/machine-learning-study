# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/1/29 13:59
# Description:
import numpy as np
from common.load_data import load_data
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score


def custom_k_fold(X_train, y_train, pipeline):
    # k折交叉验证
    kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)
    scores = []
    for k, (train, test) in enumerate(kfold):
        pipeline.fit(X_train[train], y_train[train])
        score = pipeline.score(X_train[test], y_train[test])
        scores.append(score)
        print(f"Fold: {k + 1:02d},"
              f"Class distr: {np.bincount(y_train[train])}, "
              f"Acc: {score:.3f}")
    # 计算平均得分情况
    mean_acc = np.mean(scores)
    std_acc = np.std(scores)
    print(f"CV: accuracy: {mean_acc:.3f} +/- {std_acc:.3f}")


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    pipeline = make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression())
    # sklearn 自带的k折交叉验证
    # n_jobs=-1 表示使用所有的cpu
    scores = cross_val_score(estimator=pipeline, X=X_train, y=y_train, cv=10, n_jobs=-1)
    print(f"accuracy score:  {scores}")
    print(f"CV: accuracy: {np.mean(scores):.3f} +/- "
          f"{np.std(scores):.3f}")
