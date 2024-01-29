# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/1/29 18:51
# Description: 处理类别不均衡问题
import numpy as np

from common.load_data import load_data
from sklearn.utils import resample

X_train, X_test, y_train, y_test = load_data()

# 构建类别不均衡数据集
X_imb = np.vstack((X_train[y_train == 0], X_train[y_train == 1][:40]))
y_imb = np.hstack((y_train[y_train == 0], y_train[y_train == 1][:40]))
# 验证不进行模型评估情况下准确率情况 全为0 准确率就高达87.692
y_pred = np.zeros(y_imb.shape[0])
print(np.mean(y_pred == y_imb) * 100)

print("Number of class 1 examples before: ", X_imb[y_imb == 1].shape[0], ", class 0 before: ",
      X_imb[y_imb == 0].shape[0])

# 将 X_imb[y_imb == 1], y_imb[y_imb == 1] 按照 某种规则扩充样本数量为n_samples指定的数量
# strata 用于指定样本中的分层变量 以便在重采样中根据该变量进行分层
X_resampled, y_resampled = resample(X_imb[y_imb == 1], y_imb[y_imb == 1], replace=True,
                                    n_samples=X_imb[y_imb == 0].shape[0], random_state=123)
print("Number of class 1 examples after: ", X_resampled.shape[0])
# 拼接起来
X_bal = np.vstack((X_train[y_train == 0], X_resampled))
y_bal = np.hstack((y_train[y_train == 0], y_resampled))
# 这样正负样本准确率只为50%
print("Now samples shape: ", X_bal.shape)
