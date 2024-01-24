# 感知机实现
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pandas import DataFrame
from matplotlib.colors import ListedColormap
from model.AdalineGD import AdalineGD
from model.AdalineSGD import AdalineSGD


def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x=X[y == c1, 0], y=X[y == c1, 1], alpha=0.8, c=colors[idx], marker=markers[idx],
                    label=f'Class {c1}', edgecolors='black')


def show_flower_info(df: DataFrame):
    X = df.iloc[0:100, [0, 2]].values
    # marker='o' 表示圆圈 's'表示方框 cmap
    plt.scatter(X[:50, 0], X[:50, 1], cmap=cm.get_cmap('cool'), marker='o', label='Setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], cmap=cm.get_cmap('cool'), marker='s', label='Versicolor')
    plt.xlabel("Sepal length(cm)")
    plt.ylabel("Petal length(cm)")
    plt.legend(loc="upper left")
    plt.show()


def show_model(model):
    model.fit(X_std, y)
    plot_decision_regions(X_std, y, classifier=model)

    plt.title("Adaline - Gradient descent")
    plt.xlabel("Sepal length ")
    plt.ylabel("Petal length")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

    plt.plot(range(1, len(model.losses_) + 1), model.losses_, marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Mean squared error")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    data_path = "../data/iris.data"
    print("from data path ", data_path)
    df = pd.read_csv(data_path, header=None, encoding='utf-8')
    print(df.shape)
    show_flower_info(df)

    X = df.iloc[0:100, [0, 2]].values
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', 0, 1)

    X_std = np.copy(X)
    # 数据标准化 (x - mean) / std
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    ada_gd = AdalineGD(n_iter=20, eta=0.5)
    ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
    show_model(ada_sgd)
