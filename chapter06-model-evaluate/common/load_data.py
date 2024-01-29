# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/1/29 15:41
# Description:

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_data():
    df = pd.read_csv("../data/wdbc.data", header=None)
    # 第一列是序号 不参与模型评估
    X = df.iloc[:, 2:].values
    y = df.loc[:, 1].values
    label = LabelEncoder()
    y = label.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
    return X_train, X_test, y_train, y_test
