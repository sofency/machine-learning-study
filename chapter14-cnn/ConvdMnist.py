# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/2/26 19:44
# Description:
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision
from torch.utils.data import Subset, DataLoader
import matplotlib.pyplot as plt

# 图片的大小为28 * 28 灰色图片 一个通道
image_path = "../chapter13-master-pytorch"
# transforms.ToTensor()会自动将图像数据从[0, 255]的整数像素值缩放到[0, 1]的浮点数范围
transform = transforms.Compose([transforms.ToTensor()])

mnist_dataset = torchvision.datasets.MNIST(root=image_path, train=True, transform=transform, download=True)

# 前10000为验证集
mnist_valid_dataset = Subset(mnist_dataset, torch.arange(10000))
# 10000,len(mnist_dataset) 区间为训练集
mnist_train_dataset = Subset(mnist_dataset, torch.arange(10000, len(mnist_dataset)))
# 测试集数据
mnist_test_dataset = torchvision.datasets.MNIST(root=image_path, train=False, transform=transform, download=False)

batch_size = 64
torch.manual_seed(1)
# 封装批次数据加载器
train_dl = DataLoader(mnist_train_dataset, batch_size, shuffle=True)
valid_dl = DataLoader(mnist_valid_dataset, batch_size, shuffle=True)

model = nn.Sequential()
# conv2d 默认stride为1
# 28 * 28 * 1 -> 28 * 28 * 32
model.add_module(
    "conv1", nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
)
model.add_module("relu1", nn.ReLU())
# stride 默认和kernel_size一样
# 28 * 28 * 32 -> 14 * 14 * 32
model.add_module("pool1", nn.MaxPool2d(kernel_size=2))
# 14 * 14 * 32 -> 14 * 14 * 64
model.add_module(
    "conv2",
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
)
model.add_module("relu2", nn.ReLU())
# 14 * 14 * 64 -> 7 * 7 * 64
model.add_module("pool2", nn.MaxPool2d(kernel_size=2))

# 设置全连接层
model.add_module('flatten', nn.Flatten())
# 这样可以知道flatten后是多少 即下面的3136 是如何计算出来的
# x = torch.ones((64, 1, 28, 28))
# print(model(x).shape)
model.add_module("fc1", nn.Linear(3136, 1024))
model.add_module("relu3", nn.ReLU())
model.add_module("dropout", nn.Dropout(p=0.5))
model.add_module("fc2", nn.Linear(1024, 10))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 定义以下函数训练模型
def train(model, num_epochs, train_dl, valid_dl):
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs

    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_dl:
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()

            optimizer.step()
            # 清除之前算的梯度
            optimizer.zero_grad()
            # 将所有的损失相加
            loss_hist_train[epoch] += loss.item() * y_batch.size(0)
            is_correct = (torch.argmax(y_pred, dim=1) == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.sum()

        # 计算平均的损失
        loss_hist_train[epoch] /= len(train_dl.dataset)
        # 平均准确率
        accuracy_hist_train[epoch] /= len(train_dl.dataset)

        model.eval()

        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss_hist_valid[epoch] += loss.item() * y_batch.size(0)
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
                accuracy_hist_valid[epoch] += is_correct.sum()

            loss_hist_valid[epoch] /= len(valid_dl.dataset)
            accuracy_hist_valid[epoch] /= len(valid_dl.dataset)
            print(f"Epoch {epoch + 1} train accuracy: {accuracy_hist_train[epoch]:.4f} ,"
                  f"valid accuracy: {accuracy_hist_valid[epoch]:.4f}")

    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid


def plot_image(hist):
    x_arr = np.arange(len(hist[0])) + 1
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x_arr, hist[0], '-o', label="Train loss")
    ax.plot(x_arr, hist[1], '--<', label="Validation loss")
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Loss', size=15)

    ax.legend(fontsize=15)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(x_arr, hist[2], '-o', label="Train Acc")
    ax.plot(x_arr, hist[3], '--<', label="Validation Acc")
    ax.legend(fontsize=15)

    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Accuracy', size=15)
    plt.show()


# num_epochs = 20
# hist = train(model, num_epochs, train_dl, valid_dl)
# plot_image(hist)

# 测试数据集
# mnist_test_dataset.data.shape 为 10000, 28, 28
# unsqueeze(1) 增加一个维度 10000, 1, 28, 28  然后 / 255. 归一化

# pred = model(mnist_test_dataset.data.unsqueeze(1) / 255.)

def pred_show_image(model, mnist_test_dataset):
    fig = plt.figure(figsize=(12, 4))
    for i in range(12):
        ax = fig.add_subplot(2, 6, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        # 获取第i个图片的像素部分0, [0,:,:] 获取后两个维度
        img = mnist_test_dataset[i][0][0, :, :]
        # img 增加两个维度
        pred = model(img.unsqueeze(0).unsqueeze(1))
        y_pred = torch.argmax(pred)
        ax.imshow(img, cmap='gray_r')
        ax.text(0.9, 0.1, y_pred.item(), size=15, color='blue', horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)

    plt.show()



