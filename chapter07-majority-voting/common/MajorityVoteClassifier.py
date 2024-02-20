# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/1/31 14:09
# Description:
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import _name_estimators

import numpy as np


class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers, vote="classlabel", weights=None):
        # 模型集合器
        self.classifiers_ = None
        # 样本总共几种类别
        self.classes_ = None
        # 标签编码器 将字符串变为数字
        self.label_encoder = LabelEncoder()

        self.classifiers = classifiers
        self.named_classifiers = {
            key: value for key, value in _name_estimators(classifiers)
        }
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        if self.vote not in ("probablity", "classlabel"):
            raise ValueError(f"vote must be 'probablity or 'classlabel''; got(vote={self.vote})")
        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError(f"Number of classifiers and weights must be equal; got {len(self.weights)} weights， "
                             f"{len(self.classifiers)} classifiers")
        self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.label_encoder.transform(y))
            self.classifiers_.append(fitted_clf)

        return self

    def predict(self, X):
        if self.vote == "probability":
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            maj_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                                           axis=1,
                                           arr=predictions)
        # 转换为原来的类别
        maj_vote = self.label_encoder.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self, X):
        # len(classifiers)中模型进行评估
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        # 按照列进行计算平均值 得出每种样本的最终概率
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba

    def get_params(self, deep=True):
        if not deep:
            return super().get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step, in self.named_classifiers.items():
                for key, value in step.get_params(deep=True).items():
                    out[f'{name}_{key}'] = value
            return out
