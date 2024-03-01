# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/2/27 17:18
# Description: 序列模型 给定之前时间点之前的事件，让你预测接下来时间点的事件
import torch
import numpy as np
import utils.ShowTable as ShowTable
import torch.nn as nn
from d2l import torch as d2l

T = 1000
# 随机生成1000个数字
time = torch.arange(0, T, dtype=torch.float32)
# 生成正弦函数的Y值
y = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
x_ticks = np.arange(0, 1000, 200)
ShowTable.plot(x_ticks, y, "time", xlim=[0, 1000], figsize=(6, 3))

# 将数据映射为数据对
# y_t = x_t 和 x_t = [x_(t-T),..., x_(t-1)]
tau = 4
features = torch.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = y[i:T - tau + i]

# 标签 即上述每一对数据接下来的数据是什么
labels = y[tau:].reshape((-1, 1))
batch_size, n_train = 16, 600
train_iter = ShowTable.load_data(features[:n_train], labels[:n_train], batch_size=batch_size, is_shuffle=True)


def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def get_net():
    net = nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 1))
    net.apply(init_weight)
    return net


loss_fn = nn.MSELoss()


def train(net, train_iter, loss_fn, epochs, lr):
    optimizer = torch.optim.Adam(net.parameters(), lr)
    loss_hist_train = [0] * epochs
    for epoch in range(epochs):
        for X, y in train_iter:
            loss = loss_fn(net(X), y)
            loss_hist_train[epoch] += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f'Epoch {epoch} loss {loss_hist_train[epoch] / len(train_iter):.4f}')


net = get_net()
train(net, train_iter, loss_fn, 5, 0.01)
# detach()函数会从当前计算图中分离张量，并阻止PyTorch自动计算关于这个张量的梯度。这在你只需要张量的值，而不需要进行反向传播时非常有用
onestep_pred = net(features)
ShowTable.plot(x_ticks, [y, torch.from_numpy(onestep_pred.detach().numpy())], x_label="time", xlim=[0, 1000],
               legend=["data", "one step"], figsize=(6, 4))
# 假设数据从600开始往后的每个数据都是使用自己模型预测的前4组数据作为标签的话，那么模型的表现则很不好
