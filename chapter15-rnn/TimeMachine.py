# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/2/29 17:03
# Description:
import math

import torch
from d2l import torch as d2l
import torch.nn as nn

batch_size, num_steps = 32, 35
# train_iter 每一批次32个数字，每组数据中由35个序列数据，序列数据特征和标签的差别是target 相对于feature 往后移动了一位
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)


# for x_batch, target_batch in train_iter:
#     print(x_batch.shape, target_batch.shape)
#     print(x_batch[:2, :])
#     print(target_batch[:2, :])
#     break

def get_params(vocab_size, num_hidden, device):
    # 词元 即文本中多少不重复的单词
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    W_xh = normal((num_inputs, num_hidden))
    W_hh = normal((num_hidden, num_hidden))
    b_h = torch.zeros(num_hidden, device=device)
    W_hq = normal((num_hidden, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)

    return params


def init_rnn_state(batch_size, num_hidden, device):
    # 给一批次中的数据设定y值
    return (torch.zeros((batch_size, num_hidden), device=device),)


def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # inputs [时间的步数, 批量大小, vocab_size]
    # 每一时刻的 数据进行计算
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)


#
class RNNModelScratch:
    def __init__(self, vocab_size, num_hidden, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hidden = vocab_size, num_hidden
        # 获取参数
        self.params = get_params(vocab_size, num_hidden, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    # 也可以是forward函数
    # X 就是从时间机器中load出来的数据 batch_size num_steps
    def __call__(self, X, state):
        # num_steps, batch_size, vocal_size
        X = torch.F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hidden, device)


num_hidden = 512
net = RNNModelScratch(len(vocab), num_hidden, d2l.try_gpu(), get_params, init_rnn_state, rnn)


def predict_ch8(prefix, num_preds, net, vocab, device):  # @save
    """在prefix后面生成新字符"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    # 就是将output最后一个字符作为预测下一个字符的输入
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    # 自己设定的前缀
    for y in prefix[1:]:  # 预热期
        # 没必要存储预测的字符，只需要记录对应的状态即可
        _, state = net(get_input(), state)
        # 存储前缀信息
        outputs.append(vocab[y])
    # 预测num_pred
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    # 把索引转化为字符
    return ''.join([vocab.idx_to_token[i] for i in outputs])


# 可以预测一下
# predict_ch8("time traveller ", 10, net, vocab, d2l.try_gpu())


def grad_clipping(net, theta):  # @save
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 训练损失之和,词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                state.detach_()
            else:
                # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了mean函数
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """训练模型（定义见第8章）"""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))


num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())
