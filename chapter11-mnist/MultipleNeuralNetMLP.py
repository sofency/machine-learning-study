# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/2/23 08:52
# Description: 多层感知机
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# fetch_openml 函数从openml网站下载数据集
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X.values
y = y.values.map(int, na_action='ignore')

# 将数据集划分为训练集(55000)，验证集(5000)和测试集(10000)
# stratify 参数的作用是根据指定的标签（通常是目标变量或类别标签）进行分层抽样，确保训练集和测试集中的标签比例相同
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=10000, random_state=123, stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=5000, random_state=123, stratify=y_temp)


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


# 独热编码
def int_to_onehot(y, num_labels):
    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ary[i, val] = 1
    return ary


class NeuralNetMLP:
    def __init__(self, num_features, num_hidden, num_classes, random_seed=123):
        super().__init__()
        self.num_classes = num_classes

        rng = np.random.RandomState(random_seed)
        # 隐藏层
        self.weight_h = rng.normal(loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)

        self.weight_out = rng.normal(loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)

    # 即预测结果
    def forward(self, x):
        # (n_samples, features) * (num_features, num_hidden) = (n_samples, num_hidden)
        z_h = np.dot(x, self.weight_h.T) + self.bias_h
        a_h = sigmoid(z_h)
        # (n_samples, num_hidden) * (num_hidden, num_classes) = (n_samples, num_classes)
        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        # a_out 输出的是概率可以转换为我们需要的类别标签
        a_out = sigmoid(z_out)
        return a_h, a_out

    def backward(self, x, a_h, a_out, y):
        # 独热编码
        y_onehot = int_to_onehot(y, self.num_classes)
        # loss = 1/2 * (y - y')^2  注意y是 独热编码向量 只有真实类别的索引位置为1 其他位置为0 y'为激活函数处理过的值 在-1～1之间
        # 损失函数对激活函数求偏导
        d_loss__d_a_out = (a_out - y_onehot) / y.shape[0]
        # 激活函数求偏导所得
        d_a_out__d_z_out = a_out * (1. - a_out)
        delta_out = d_loss__d_a_out * d_a_out__d_z_out
        d_z_out__dw_out = a_h
        d_loss__dw_out = np.dot(delta_out.T, d_z_out__dw_out)
        d_loss__db_out = np.sum(delta_out, axis=0)

        # 计算隐藏层
        d_z_out__a_h = self.weight_out
        d_loss__a_h = np.dot(delta_out, d_z_out__a_h)

        d_a_h__d_z_h = a_h * (1 - a_h)
        d_z_h__d_w_h = x
        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T, d_z_h__d_w_h)

        d_loss__d_b_h = np.sum((d_loss__a_h * d_a_h__d_z_h), axis=0)
        return (d_loss__dw_out, d_loss__db_out,
                d_loss__d_w_h, d_loss__d_b_h)


model = NeuralNetMLP(num_features=28 * 28, num_hidden=50, num_classes=10)
num_epochs = 50
minibatch_size = 100


# 生成小批量数据
def minibatch_generator(X, y, minibatch_size):
    indices = np.arange(X.shape[0])
    # 将索引打乱顺序
    np.random.shuffle(indices)
    for start_idx in range(0, indices.shape[0] - minibatch_size + 1, minibatch_size):
        batch_idx = indices[start_idx:start_idx + minibatch_size]
        yield X[batch_idx], y[batch_idx]


def mse_loss(targets, probas, num_labels=10):
    onehot_targets = int_to_onehot(targets, num_labels=num_labels)
    return np.mean((onehot_targets - probas) ** 2)


def accuracy(targets, predicted_labels):
    return np.mean(predicted_labels == targets)


_, probas = model.forward(X_valid)
mse = mse_loss(y_valid, probas)
print(f"Initial validation MSE: {mse:.1f}")
predict_labels = np.argmax(probas, axis=1)
acc = accuracy(y_valid, predict_labels)
print(f"Initial validation accuracy : {acc * 100:.2f}%")


#
def compute_mse_and_acc(nnet, X, y, num_labels=10, minibatch_size=100):
    mse, correct_pred, num_examples = 0., 0, 0
    minibatch_gen = minibatch_generator(X, y, minibatch_size)
    count = 0
    for i, (features, targets) in enumerate(minibatch_gen):
        # 预测结果
        _, probas = nnet.forward(features)
        # 找出类别中可能性最大的那个
        predict_labels = np.argmax(probas, axis=1)
        # 将目标进行独热编码
        onehot_targets = int_to_onehot(targets, num_labels=num_labels)
        # 计算误差
        loss = np.mean((onehot_targets - probas) ** 2)
        # 预测正确的个数
        correct_pred += (predict_labels == targets).sum()
        num_examples += targets.shape[0]
        mse += loss
        count += 1
    # /n
    mse = mse / count
    acc = correct_pred / num_examples
    return mse, acc


def train(model, X_train, y_train, X_valid, y_valid, num_epochs, learning_rate=0.1):
    epoch_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []

    for e in range(num_epochs):
        minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)
        for X_train_mini, y_train_mini in minibatch_gen:
            # 预测
            a_h, a_out = model.forward(X_train_mini)
            # 反向传播更新参数
            d_loss__dw_out, d_loss__db_out, d_loss__d_w_h, d_loss__d_b_h = \
                model.backward(X_train_mini, a_h, a_out, y_train_mini)

            model.weight_h -= learning_rate * d_loss__d_w_h
            model.bias_h -= learning_rate * d_loss__d_b_h
            model.weight_out -= learning_rate * d_loss__dw_out
            model.bias_out -= learning_rate * d_loss__db_out

        # 计算均方差和准确率
        train_mse, train_acc = compute_mse_and_acc(model, X_train, y_train)
        valid_mse, valid_acc = compute_mse_and_acc(model, X_valid, y_valid)
        train_acc, valid_acc = train_acc * 100, valid_acc * 100
        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(valid_acc)
        epoch_loss.append(train_mse)
        print(f'Epoch: {e + 1:03d}/{num_epochs:03d} '
              f'| Train MSE: {train_mse:.2f} '
              f'| Train ACC: {train_acc:.2f}% '
              f'| Valid ACC: {valid_acc:.2f}%')
    return epoch_loss, epoch_train_acc, epoch_valid_acc


np.random.seed(123)

epoch_loss, epoch_train_acc, epoch_valid_acc = train(model, X_train, y_train, X_valid, y_valid, num_epochs=50,
                                                     learning_rate=0.1)
plt.plot(range(len(epoch_train_acc)), epoch_train_acc, label="Train Acc")
plt.plot(range(len(epoch_valid_acc)), epoch_valid_acc, label="Valid Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.show()

# 计算测试集
test_mse, test_acc = compute_mse_and_acc(model, X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
