import numpy as np


class AdalineGD:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        """
        :param eta: 学习率
        :param n_iter: 迭代次数
        :param random_state: 随机数种子
        """
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)  # 随机数种子
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])  # 生成均值为0，方差为0.01的正态分布随机数
        self.b_ = np.float_(0.)
        self.losses_ = []

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors ** 2).mean()
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        return X

    def predict(self, X):
        # np.where类似与三元运算符 self.net_input(X) >= 0.0 ? 1 : 0
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
