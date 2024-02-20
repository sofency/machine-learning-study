# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/2/20 10:35
# Description:
import random

import numpy as np
import pandas as pd
from sklearn import datasets


def load_data():
    # 返回的数据是已经加工过的，没有字符串
    iris = datasets.load_iris()
    return iris.data, iris.target


def normalization_result(data):
    data = data - min(data)
    data = data / max(data)
    res = []
    for item in data:
        if item <= 0.33:
            res.append(0)
        elif item <= 0.66:
            res.append(1)
        else:
            res.append(2)
    res = np.array(res)
    return res


def predict(X, res_k):
    mis = (res_k * X).sum(axis=1)
    return normalization_result(mis)


def train(X, y, w):
    # 保存加权错误率
    min_loss = 1
    features = X.shape[1]
    res_k = None
    nums = 1
    for t in range(nums):
        k_random = []
        for i in range(features):
            k_random.append(random.uniform(0, 1))
        k_random = np.array(k_random)
        res = predict(X, k_random)
        # 计算加权错误率
        miss = sum((res != y) * w)
        if miss < min_loss:
            min_loss = miss
            res_k = k_random
    return min_loss, res_k


def adaboost(data, target, iters):
    # 样本数
    samples = data.shape[0]
    # 初始化权重向量w使其所有元素非负且数值相同
    w_m = [1 / samples] * samples
    res = np.zeros(samples)
    # 迭代次数
    for m in range(iters):
        # 获得弱学习器
        min_loss, res_k = train(data, target, w_m)
        # 计算系数
        a = 1 / 2 * np.log((1 - min_loss) / min_loss)
        # 预测列别标签
        y_pred = predict(data, res_k)
        # 更新权重
        w_m = w_m * np.exp(-a * y_pred * target)
        # 归一化w
        w_m = w_m / np.sum(w_m)
        res += a * y_pred
    # 归一化
    result = normalization_result(res)
    percent = sum(result == target) / samples
    print("正确率 {}%".format(percent * 100))


data, target = load_data()
adaboost(data, target, 4)
