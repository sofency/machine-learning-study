# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/2/18 16:45
# Description:
import numpy as np
from itertools import product

if __name__ == '__main__':
    array = np.array([[1, 2, 3, 4],
                      [2, 3, 4, 5]])
    # 按照列进行计算
    print(np.sum(array, axis=0))
    # 按照行进行计算
    print(np.sum(array, axis=1))
    # 生成坐标位置
    for arr in product([0, 1], [0, 1]):
        print(arr[0], arr[1])

    X = np.array([[1, 2, 3, 4], [3, 4, 3, 22], [1, 1, 1, 1]])
    w = np.array([0.1, 0.2, 0.3, 0.4])
    res = w * X
    print(res)
    print(res.sum(axis=1))  # w1*x1 + w2 * x2结果是用来判断类别的
