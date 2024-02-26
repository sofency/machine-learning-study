# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/2/25 17:43
# Description: 自定义Pytorch的网络层，需要继承nn.Module
import torch.nn as nn
import torch


class NoisyLinear(nn.Module):
    def __init__(self, input_size, output_size, noise_stddev=0.1):
        super().__init__()
        w = torch.Tensor(input_size, output_size)
        self.w = nn.Parameter(w)
        nn.init.xavier_uniform_(self.w)
        # 填充0作为初始b值
        b = torch.Tensor(output_size).fill_(0)
        self.b = nn.Parameter(b)
        self.noise_stddev = noise_stddev

    def forward(self, x, training=False):
        if training:
            noise = torch.normal(0.0, self.noise_stddev, x.shape)
            x_new = torch.add(x, noise)
        else:
            x_new = x
        # w*(x+noise)+b
        return torch.add(torch.mm(x_new, self.w), self.b)


# 如果forward中没有training这个参数 可以使用nn.Sequential连接起来模型
class MyNoisyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.noisy = NoisyLinear(2, 4, 0.07)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(4, 4)
        self.activation2 = nn.ReLU()
        self.linear3 = nn.Linear(4, 1)
        self.activation3 = nn.Sigmoid()

    def forward(self, x, training=False):
        x = self.noisy(x, training)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        x = self.activation3(x)
        return x

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        pred = self.forward(x)[:, 0]
        return (pred >= 0.5).float()


