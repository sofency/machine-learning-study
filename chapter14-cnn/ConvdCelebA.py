# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/2/27 09:33
# Description: 使用卷积神经网络对人脸图像进行微笑分析
# 原始数据集图片大小为 178×218
import torch
from torchvision.transforms import transforms
import torchvision
import matplotlib.pyplot as plt
from torchvision.transforms.functional import *
from torch.utils.data import Subset, DataLoader
import torch.nn as nn


# 图片转换和数据增广
def plot_ten_image_after_deal(celebA_train_dataset):
    fig = plt.figure(figsize=(16, 8.5))
    ax = fig.add_subplot(2, 5, 1)
    # attr里面是是否微笑 以及年龄
    img, attr = celebA_train_dataset[0]
    ax.set_title("Crop to a \nbounding-box", size=15)
    ax.imshow(img)
    ax = fig.add_subplot(2, 5, 6)
    # top left  height width 四个坐标确定裁剪的图片范围
    img_cropped = crop(img, 50, 20, 128, 128)
    ax.imshow(img_cropped)

    # 水平反转
    ax = fig.add_subplot(2, 5, 2)
    img, attr = celebA_train_dataset[1]
    ax.set_title("Flip (horizontal)", size=15)
    ax.imshow(img)
    ax = fig.add_subplot(2, 5, 7)
    # hflip 水平反转
    img_flipped = hflip(img)
    #
    ax.imshow(img_flipped)

    # 调整对比度
    ax = fig.add_subplot(2, 5, 3)
    img, attr = celebA_train_dataset[2]
    ax.set_title("Adjust contrast)", size=15)
    ax.imshow(img)
    ax = fig.add_subplot(2, 5, 8)
    img_adj_contrast = adjust_contrast(img, contrast_factor=2)
    ax.imshow(img_adj_contrast)

    # 调整亮度
    ax = fig.add_subplot(2, 5, 4)
    img, attr = celebA_train_dataset[3]
    ax.set_title("Adjust Brightness)", size=15)
    ax.imshow(img)
    ax = fig.add_subplot(2, 5, 9)
    img_adj_brightness = adjust_brightness(img, brightness_factor=1.3)
    ax.imshow(img_adj_brightness)

    # 裁剪图片中心图片并将生成的图像调整为原始大小
    ax = fig.add_subplot(2, 5, 5)
    img, attr = celebA_train_dataset[3]
    ax.set_title("Center crop\nand resize", size=15)
    ax.imshow(img)
    ax = fig.add_subplot(2, 5, 10)
    img_center_crop = center_crop(img, [0.7 * 218, 0.7 * 218])
    image_resize = resize(img_center_crop, size=(218, 218))
    ax.imshow(image_resize)


# 训练模型时只针对训练数据使用数据增广方法，在模型的验证或测试阶段不使用数据增广方法
transform_train = transforms.Compose([
    transforms.RandomCrop([178, 178]),
    transforms.RandomHorizontalFlip(),
    transforms.Resize([64, 64]),
    transforms.ToTensor()
])

# 用于验证集和测试集
transform = transforms.Compose([
    transforms.CenterCrop([178, 178]),
    transforms.Resize([64, 64]),
    transforms.ToTensor()
])

image_path = "./"
# 下载训练集，验证集，测试集
# 获取是否微笑
# img.permute(1, 2, 0) 会将原始张量的维度从（通道，高度，宽度）重新排列为（高度，宽度，通道）。
get_smile = lambda attr: attr[31]
celebA_train_dataset = torchvision.datasets.CelebA(image_path, split='train', target_type='attr', download=True,
                                                   transform=transform_train, target_transform=get_smile)
celebA_valid_dataset = torchvision.datasets.CelebA(image_path, split='valid', target_type='attr', download=True,
                                                   transform=transform, target_transform=get_smile)

celebA_test_dataset = torchvision.datasets.CelebA(image_path, split='test', target_type='attr', download=True,
                                                  transform=transform, target_transform=get_smile)

# 使用一部分进行训练模型
celebA_train_dataset = Subset(celebA_train_dataset, torch.arange(16000))
celebA_valid_dataset = Subset(celebA_valid_dataset, torch.arange(1000))

# 创建数据加载器
batch_size = 32
train_dl = DataLoader(celebA_train_dataset, batch_size, shuffle=True)
valid_dl = DataLoader(celebA_valid_dataset, batch_size, shuffle=True)
test_dl = DataLoader(celebA_test_dataset, batch_size, shuffle=True)

# 训练卷积圣经网络分类器
model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Dropout(p=0.5),
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Dropout(p=0.5),
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
    nn.ReLU(),
    # 上述卷积层的输出为 torch.Size([batch_size, 256, 8, 8])
    nn.AvgPool2d(kernel_size=8),
    # 变为[batch_size,256]
    nn.Flatten(),
    nn.Linear(256, 1),  # [batch_size,1]
    nn.Sigmoid()
)

# 定义损失函数
loss_fn = nn.BCELoss()
# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 定义训练模型
def train(model, num_epochs, train_dl, valid_dl):
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs

    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_dl:
            # 单标签分类问题需要加上，多分类问题则不需要
            pred = model(x_batch)[:, 0]
            loss = loss_fn(pred, y_batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_hist_train[epoch] += loss.item()
            is_correct = ((pred > 0.5).float() == y_batch).float().sum()
            accuracy_hist_train[epoch] += is_correct

        loss_hist_train[epoch] /= len(train_dl.dataset)
        accuracy_hist_train[epoch] /= len(train_dl.dataset)

        model.eval()

        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                pred = model(x_batch)[:, 0]
                loss = loss_fn(pred, y_batch)
                loss_hist_valid[epoch] += loss.item()
                is_correct = ((pred > 0.5).float() == y_batch).float().sum()
                accuracy_hist_valid[epoch] += is_correct

            loss_hist_valid[epoch] /= len(valid_dl.dataset)
            accuracy_hist_valid[epoch] /= len(valid_dl.dataset)

        print(f'Epoch {epoch + 1}: accuracy: {accuracy_hist_train[epoch]:.4f} '
              f'valid accuracy: {accuracy_hist_valid[epoch]:.4f}')
    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid
