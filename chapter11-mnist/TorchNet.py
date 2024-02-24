# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/2/24 11:00
# Description: 在Pytorch中构建神经网络模型
import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

iris = load_iris()
X = iris["data"]
y = iris["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# 每一列
X_train_norm = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
X_train_norm = torch.from_numpy(X_train_norm).float()
y_train = torch.from_numpy(y_train)

train_ds = TensorDataset(X_train_norm, y_train)
torch.manual_seed(1)
batch_size = 2
train_dl = DataLoader(train_ds, batch_size, shuffle=True)


class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = torch.nn.Linear(input_size, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.Sigmoid()(x)
        x = self.layer2(x)
        return x


# 即4个
input_size = X_train_norm.shape[1]
hidden_size = 16
output_size = 3

model = Model(input_size, hidden_size, output_size)
learning_rate = 0.001
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 100
loss_hist = [0] * num_epochs
accuracy_hist = [0] * num_epochs

for epoch in range(num_epochs):
    # 一批就两个
    for x_batch, y_batch in train_dl:
        pred = model.forward(x_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_hist[epoch] += loss.item() * y_batch.size(0)
        is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
        accuracy_hist[epoch] += is_correct.sum()

    loss_hist[epoch] /= len(train_dl.dataset)
    accuracy_hist[epoch] /= len(train_dl.dataset)

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1)
ax.plot(loss_hist, lw=3)
ax.set_title("Training Loss", size=15)
ax.set_xlabel("Epoch", size=15)
ax.tick_params(axis='both', which='major', labelsize=15)

ax = fig.add_subplot(1, 2, 2)
ax.plot(accuracy_hist, lw=3)
ax.set_title("Training Accuracy", size=15)
ax.set_xlabel("Epoch", size=15)
ax.tick_params(axis='both', which='major', labelsize=15)

plt.show()

# 在测试集上测试数据
# 处理测试数据
X_test_norm = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)
X_test_norm = torch.from_numpy(X_test_norm).float()
y_test = torch.from_numpy(y_test)
pred_test = model.forward(X_test_norm)
correct = (torch.argmax(pred_test, dim=1) == y_test).float()
print(correct)
accuracy = correct.sum() / len(X_test_norm)
print(f'Test Acc: {accuracy:.4f}')

# 将训练好的模型参数保存
path = "iris_classifier.pt"
torch.save(model, path)

# model_new = torch.load(path) 加载模型参数生成新的模型 然后进行吧预测

