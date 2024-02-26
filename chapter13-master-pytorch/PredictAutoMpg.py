# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/2/25 18:13
# Description: 汽车燃油效率预测
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.nn.functional import one_hot
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

column_names = ["MPG", "Cylinders", "Displacement", "Horsepower", "Weight", "Acceleration", "Model Year", "Origin"]
# skipinitialspace 参数可以帮助你处理 CSV 文件中字段值前面的不必要空白字符
df = pd.read_csv("../data/auto-mpg.data", names=column_names, na_values='?', comment='\t', sep=' ',
                 skipinitialspace=True)

# 因为缺少的数据只有6行 对于398个样本数据来说，比较少，因此可以删除
# 使用drop=True 则原来的索引列信息则不再使用，否则还会使用原来的索引列，即数据会增加一列索引列
df = df.dropna()
df = df.reset_index(drop=True)

df_train, df_test = train_test_split(df, train_size=0.8, random_state=1)
# 描述信息转置
# 之所以使用训练数据的信息，是为了保持数据分布的一致性
# 标准化步骤
# 1.计算训练数据的均值（mean）和标准差（standard deviation）。
# 2.使用训练数据的均值和标准差来标准化训练数据。
# 3.使用相同的均值和标准差来标准化测试数据。
train_stats = df_train.describe().transpose()

numeric_column_names = [
    "Cylinders", "Displacement", "Horsepower", "Weight", "Acceleration"
]
df_train_norm, df_test_norm = df_train.astype(float).copy(), df_test.astype(float).copy()

for col_name in numeric_column_names:
    mean = train_stats.loc[col_name, 'mean']
    std = train_stats.loc[col_name, 'std']
    df_train_norm.loc[:, col_name] = (df_train_norm.loc[:, col_name] - mean) / std
    df_test_norm.loc[:, col_name] = (df_test_norm.loc[:, col_name] - mean) / std

# 边界分组
boundaries = torch.tensor([73, 76, 79])
v = torch.tensor(df_train_norm['Model Year'].values)
df_train_norm['Model Year Bucketed'] = torch.bucketize(v, boundaries, right=True)

v = torch.tensor(df_test_norm['Model Year'].values)
df_test_norm['Model Year Bucketed'] = torch.bucketize(v, boundaries, right=True)
numeric_column_names.append("Model Year Bucketed")

# 没有正则化处理的
total_origin = len(set(df_train_norm['Origin']))
origin_encoded = one_hot(torch.from_numpy(df_train_norm['Origin'].values.astype(int)) % total_origin)
# 转化为tensor
x_train_numeric = torch.tensor(df_train_norm[numeric_column_names].values)
x_train = torch.cat([x_train_numeric, origin_encoded], 1).float()

origin_encoded = one_hot(torch.from_numpy(df_test_norm['Origin'].values.astype(int)) % total_origin)
# 转化为tensor
x_test_numeric = torch.tensor(df_test_norm[numeric_column_names].values)
x_test = torch.cat([x_test_numeric, origin_encoded], 1).float()

y_train = torch.tensor(df_train_norm['MPG'].values).float()
y_test = torch.tensor(df_test_norm['MPG'].values).float()

# 训练DNN回归模型
train_ds = TensorDataset(x_train, y_train)
batch_size = 8
torch.manual_seed(1)
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# 构建模型
model = nn.Sequential(
    nn.Linear(x_train.shape[1], 8),
    nn.ReLU(),
    nn.Linear(8, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

num_epochs = 200
log_epochs = 20
for epoch in range(num_epochs):
    loss_hist_train = 0
    for x_batch, y_batch in train_dl:
        y_pred = model(x_batch)[:, 0]
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_hist_train += loss.item()
    if epoch % log_epochs == 0:
        print(f'Epoch {epoch} Loss: {loss_hist_train / len(train_dl):.4f}')

with torch.no_grad():
    test_pred = model(x_test)[:, 0]
    loss = loss_fn(test_pred, y_test)
    print(f'Test MSE: {loss.item():.4f}')



