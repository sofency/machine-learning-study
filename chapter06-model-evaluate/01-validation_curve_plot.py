# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/1/29 15:44
# Description:
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import validation_curve
import numpy as np
from common.load_data import load_data

pipeline = make_pipeline(StandardScaler(), LogisticRegression(penalty='l2', max_iter=10000))

X_train, X_test, y_train, y_test = load_data()
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(estimator=pipeline, X=X_train, y=y_train,
                                             param_name='logisticregression__C', param_range=param_range, cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label="Training accuracy")
plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')

plt.plot(param_range, test_mean, color='green', marker='s', linestyle='--', markersize=5, label="Validation accuracy")
# 负责填充上述折线附近的空间 范围是mean +- std
plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

plt.grid()
plt.xscale('log')
plt.xlabel("Number of training examples")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.ylim([0.8, 1.0])
plt.show()

