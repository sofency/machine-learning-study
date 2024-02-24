# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/2/23 15:27
# Description:
import torch
import torchvision
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
# 用于从迭代器中获取指定数量的元素。
from itertools import islice

np.set_printoptions(precision=3)
numpy_b = np.array([2, 3, 4], dtype=np.int32)

tensor_a = torch.tensor([1, 2, 3])
# 将numpy转换为对应的张量
tensor_b = torch.from_numpy(numpy_b)

# 生成2*3矩阵 元素全1的矩阵
print(torch.ones(2, 3))
# 生成随机张量
print(torch.rand(2, 3))

# 将张量转换为所需的类型
tensor_a_float = tensor_a.to(torch.float32)
print(tensor_a_float)

# 张量转置 只针对二维矩阵
# print(torch.transpose(tensor_a_float, 0, 1)) error
# 后面两个参数是对应的维度
array = torch.rand(3, 4)
print(torch.transpose(array, 0, 1))

# 张量reshape
array = array.reshape(2, 6)
print(array.shape)

# 删除不必要的维度
t = torch.zeros(1, 2, 1, 4, 1)
# 删除索引为2的对应的维度1
t_sqz = torch.squeeze(t, 2)
print(t_sqz.shape)

#
torch.manual_seed(1)
# rand是在-1，1之间随机生成
t1 = 2 * torch.rand(5, 2) - 1
print(t1)
# 生成均值为0，方差为1的（5，2）矩阵
t2 = torch.normal(mean=0, std=1, size=(5, 2))
print(t2)
# 对应位置元素相乘
t3 = torch.multiply(t1, t2)
print(t3)
print("--------")
# axis=0 是矩阵按照 每一列的元素和的均值为新值  axis=1 每一行的元素和的均值为新值
print(torch.mean(t1, axis=0))

# 矩阵相乘
print(torch.matmul(t1, torch.transpose(t2, 0, 1)))
# .T也是转置
print(torch.matmul(t1.T, t2))

# 计算张量的Lp范数
norm_t1 = torch.linalg.norm(t1, ord=2, dim=1)
print(norm_t1)

print(np.sqrt(np.sum(np.square(t1.numpy()), axis=1)))

# 拆分张量
# 生成随机6个-1～1之间的小数
tensor_c = torch.rand(6)
# 每个张量里面包含 len(tensor_c) / 3 个元素
tensor_d = torch.chunk(tensor_c, 3)
print(tensor_d)
result = [item.numpy() for item in tensor_d]
print(result)

# 自己可以设定每一份拆分的数量
tensor_e = torch.split(tensor_c, split_size_or_sections=[1, 2, 2, 1])
print(tensor_e)

A, B, B1 = torch.ones(3), torch.zeros(2), torch.zeros(3)
# 水平拼接
C = torch.cat([A, B])
print(C)
#
# print(torch.stack([A, B1])) 垂直拼接
# A和B 都转置然后向右拼接
print(torch.stack([A, B1], axis=1))

# 使用已有张量创建Pytorch DataLoader

t = torch.arange(6, dtype=torch.float32)
data_loader = DataLoader(t)
for item in data_loader:
    print(item)

# 创建大小为3的批数据
data_loader = DataLoader(t, batch_size=3, drop_last=False)
for i, batch in enumerate(data_loader):
    print(f"Batch {i + 1}:", batch)

# 张量合并
t_x = torch.rand([4, 3], dtype=torch.float32)
t_y = torch.arange(4)


# 创建DataSet类
class JointDataSet(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.y[item]


# 或者使用pytorch自带的
from torch.utils.data import TensorDataset

# joint_data_set = JointDataSet(t_x, t_y)
joint_data_set = TensorDataset(t_x, t_y)
for example in joint_data_set:
    print(f"num 0: {example[0]}, num 1: {example[1]}")

# 利用joint_data_set数据集创建一个乱序的数据加载器
# 这样就不会时特征和标签的对应关系错乱
data_loader = DataLoader(dataset=joint_data_set, batch_size=2, shuffle=True)
for i, batch in enumerate(data_loader):
    print(f'batch {i}: x:{batch[0]}, y:{batch[1]}')

imgdir_path = pathlib.Path("cat_dog_images")
print(imgdir_path)
file_dist = sorted([str(path) for path in imgdir_path.glob('*.jpg')])
print(file_dist)

# 使用matplotlib可视化图像
fig = plt.figure(figsize=(10, 5))
for i, file in enumerate(file_dist):
    img = Image.open(file)
    print("image shape: ", np.array(img).shape)
    # 貌似不允许设置行宽  表示在[2,3]矩阵的 索引i+1处设置图片
    ax = fig.add_subplot(2, 3, i + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(img)
    ax.set_title(os.path.basename(file), size=15)

plt.tight_layout()
plt.show()

# 给图片打上标签
labels = [1 if 'dog' in os.path.basename(file) else 0 for file in file_dist]
print(labels)


class ImageDataSet(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        return self.images[item], self.labels[item]


image_dataset = ImageDataSet(file_dist, labels)
for file, label in image_dataset:
    print(file, label)

# 将原始图像转化为80*120像素
img_height, img_width = 80, 120
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((img_height, img_width))
])


class ImageDataSetVersion1(Dataset):
    def __init__(self, images, labels, transforms=None):
        self.images = images
        self.labels = labels
        self.transform = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = Image.open(self.images[item])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[item]


image_dataset = ImageDataSetVersion1(file_dist, labels, transform)
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 6), sharex=True, sharey=True)
ax = ax.flatten()

for i, image in enumerate(image_dataset):
    # imshow 最后显示通道的图片信息
    ax[i].imshow(image[0].numpy().transpose(1, 2, 0), cmap='Greys')
    ax[i].set_title(image[1], size=15)

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

# 使用torchvision.datasets.下的数据集
image_path = './'
# 下载数据量较大
celebA_dataset = torchvision.datasets.CelebA(
    image_path, split='train', target_type='attr', download=True
)

fig = plt.figure(figsize=(12, 8))
# 到18结束
for i, (image, attributes) in islice(enumerate(celebA_dataset), 18):
    ax = fig.add_subplot(3, 6, i + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(image)
    ax.set_title(attributes[31], size=15)

plt.show()
