# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/2/22 10:59
# Description:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix
from mlxtend.plotting import heatmap
from sklearn.preprocessing import StandardScaler
from model.LinearRegressionGD import LinearRegressionGD

columns = ["Overall Qual", "Overall Cond", "Gr Liv Area", "Central Air", "Total Bsmt SF", "SalePrice"]
df = pd.read_csv("http://jse.amstat.org/v19n3/decock/AmesHousing.txt", sep='\t', usecols=columns)
# 查询当前列存在多少个值
print(df['Central Air'].unique())
# 编码Central Air为数值类型
df['Central Air'] = df['Central Air'].map({'Y': 1, 'N': 0})

# 检查缺失值
print(df.isnull().sum())
# 由于缺失值仅一行，缺失数量对于样本总数来说太小，所以删除该缺失样本即可
# inplace=True 直接在原df上操作
df.dropna(axis=0, inplace=True)

scatterplotmatrix(df.values, figsize=(12, 10), names=df.columns, alpha=0.5)
plt.tight_layout()
plt.show()

cm = np.corrcoef(df.values.T)
hm = heatmap(cm, row_names=df.columns, column_names=df.columns)
plt.tight_layout()
plt.show()

X = df[["Gr Liv Area"]].values
y = df["SalePrice"].values

sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()

# from sklearn.linear_model import LinearRegression
# lr = LinearRegression()
lr = LinearRegressionGD(eta=0.1)
lr.fit(X_std, y_std)

plt.plot(range(1, lr.n_iter + 1), lr.losses_)
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.show()


def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolors='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)


# 就只展示Gr Liv Area 与销售价格SalePrice之间的关系
lin_regplot(X_std, y_std, lr)
plt.xlabel(' Living area above ground (Standardized)')
plt.ylabel(" Sale price (Standardized)")
# 线性回归直线反应了房价随着地面及以上居住面积的增大而增加的总体趋势
plt.show()

