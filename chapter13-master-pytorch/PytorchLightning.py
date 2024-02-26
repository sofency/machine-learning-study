# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/2/26 10:50
# Description: Pytorch Lightning训练模式 就是将下述方法填充好就可以进行训练
import pytorch_lightning as pl
import torch
import torch.nn as nn

from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms


class MultiLayerPerceptron(pl.LightningModule):
    def __init__(self, image_shape=(1, 28, 28), hidden_units=(32, 16)):
        super().__init__()
        self.train_acc = Accuracy(task='multiclass', num_classes=10)
        self.valid_acc = Accuracy(task='multiclass', num_classes=10)
        self.test_acc = Accuracy(task='multiclass', num_classes=10)

        input_size = image_shape[0] * image_shape[1] * image_shape[2]
        all_layers = [nn.Flatten()]
        for hidden_unit in hidden_units:
            layer = nn.Linear(input_size, hidden_unit)
            all_layers.append(layer)
            all_layers.append(nn.ReLU())
            input_size = hidden_unit
        all_layers.append(nn.Linear(hidden_units[-1], 10))
        self.model = nn.Sequential(*all_layers)

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.functional.cross_entropy(y_pred, y)
        y_label = torch.argmax(y_pred, dim=1)
        self.train_acc.update(y_label, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train acc", self.train_acc.compute())

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.functional.cross_entropy(y_pred, y)
        y_label = torch.argmax(y_pred, dim=1)
        self.valid_acc.update(y_label, y)
        self.log("valid_loss", loss, prog_bar=True)
        self.log("valid acc", self.valid_acc.compute(), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.functional.cross_entropy(y_pred, y)
        y_label = torch.argmax(y_pred, dim=1)
        self.test_acc.update(y_label, y)
        self.log("test loss", loss, prog_bar=True)
        self.log("test acc", self.test_acc.compute(), prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


class MnistDataModule(pl.LightningDataModule):
    def __init__(self, data_path="./"):
        super().__init__()
        self.data_path = data_path
        self.transform = transforms.Compose([transforms.ToTensor()])

    def prepare_data(self) -> None:
        MNIST(root=self.data_path, download=True)

    def setup(self, stage: str) -> None:
        mnist_all = MNIST(root=self.data_path, train=True, transform=self.transform, download=False)
        self.train, self.val = random_split(mnist_all, [55000, 5000], generator=torch.Generator().manual_seed(1))
        self.test = MNIST(root=self.data_path, train=False, transform=self.transform, download=False)

    # 数据加载器
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=64, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=64, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.val, batch_size=64, num_workers=0)


torch.manual_seed(1)
mnist_dm = MnistDataModule()

mnist_classifier = MultiLayerPerceptron()
trainer = pl.Trainer(max_epochs=10)

# 训练
trainer.fit(model=mnist_classifier, datamodule=mnist_dm)

# 测试集上评估模型
# trainer.test(model=mnist_classifier, datamodule=mnist_dm)

# 默认情况下Lightning在名为Lightning_logs的子文件夹下存储了模型训练的跟踪信息
# tensorboard --logdir lightning_logs/

# 还可以加载最新模型的checkpoint进行训练
# trainer = pl.Trainer(max_epochs=15,
#               resume_from_checkpoint='./lightning_logs/version_4/checkpoints/epoch=9-step=8600.ckpt')
# trainer.fit(model=mnist_classifier, datamodule=mnist_dm)

# 重新使用该模型
# model = MultiLayerPerceptron.load_from_checkpoint("path/to/checkpoint.ckpt")
