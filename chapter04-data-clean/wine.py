import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression


def feature_scale(scale_type, X_train, X_test):
    scales = {
        # 特征缩放 为了避免特征值大的影响较大 需要对特征进行缩放
        # 最大最小缩放 x = (xi - xmin) / (xmax -xmin)
        "minmax": MinMaxScaler(),
        # 标准化缩放 x = (xi - mean) / std
        "std": StandardScaler()
    }
    scaler = scales.get(scale_type)
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    return X_train_norm, X_test_norm


def plot_weight_cart(X_train_std, df_wine: DataFrame, y_train):
    fig = plt.figure()
    ax = plt.subplot(111)
    colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "black", "pink", "lightgreen", "lightblue", "gray",
              "indigo", "orange"]
    weights, params = [], []
    for c in np.arange(-4, 6):
        lr = LogisticRegression(penalty='l1', C=10. ** c, solver='liblinear', multi_class='ovr', random_state=0)
        lr.fit(X_train_std, y_train)
        weights.append(lr.coef_[1])
        params.append(10. ** c)
    weights = np.array(weights)
    for column, color in zip(range(weights.shape[1]), colors):
        plt.plot(params, weights[:, column], label=df_wine.columns[column + 1], color=color)
    plt.axhline(0, color='black', linestyle='--', linewidth=3)  # 是一个用于绘制水平线的函数，表示在y=0的位置画一条线
    plt.xlim([10 ** (-5), 10 ** 5])  # 设置x轴的范围从10的-5次方到10的5次方
    plt.ylabel("Weight coefficient")
    plt.xlabel("C (inverse regularization) strength")
    plt.xscale('log')  # 设置x轴为对数刻度
    plt.legend(loc="upper left")
    ax.legend(loc="upper center", bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
    plt.show()


# 序贯特征选择算法
def select_k_feature(df_wine: DataFrame, X_train_std, y_train):
    from sklearn.neighbors import KNeighborsClassifier
    from model.CustomSequentialFeatureSelector import SequentialFeatureSelector
    knn = KNeighborsClassifier(n_neighbors=5)
    sbs = SequentialFeatureSelector(knn, k_features=1)
    sbs.fit(X_train_std, y_train)

    k_feature = [len(k) for k in sbs.subsets_]
    plt.plot(k_feature, sbs.scores_, marker='o')
    plt.ylim([0.7, 1.02])
    plt.ylabel("Accuracy")
    plt.xlabel("Number of features")
    plt.grid()
    plt.tight_layout()
    plt.show()

    # 图中看到4， 9， 10 11 准确率达到100%
    k3 = list(sbs.subsets_[10])
    print(df_wine.columns[1:][k3])


# 设定指定阀值选择特征
def select_k_model(forest, X_train, feat_labels, indices, importances):
    from sklearn.feature_selection import SelectFromModel
    # 阈值设置为0.1
    sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
    X_selected = sfm.transform(X_train)
    print("Number of features that meet this threshold", 'criterion:', X_selected.shape[1])
    for f in range(X_selected.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))


# 随机森林评估模型特征重要性
# 随机森林存在的问题 如果两个或多个特征高度相关，那么可能一个特征的排名比较高，其他的特征的信息无法完全捕捉，
# 如果只对模型预测性感兴趣，而对特征重要性解释不太感兴趣，则不必担心这个问题
def evaluate_feature_importance(df_wine: DataFrame, X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    feat_labels = df_wine.columns[1:]
    forest = RandomForestClassifier(n_estimators=500, random_state=1)
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    # 倒序
    indices = np.argsort(importances)[::-1]
    for f in range(X_train.shape[1]):
        # %*s：这是一个格式说明符，表示要打印一个字符串。星号*表示宽度是一个变量，其值由后续的参数决定。这里，宽度被设置为30（由30这个参数决定）
        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

    plt.title("Feature importance")
    plt.bar(range(X_train.shape[1]), importances[indices], align='center')
    plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    plt.show()

    # SelectFromModel 需要借助随机森林进行按照阈值选取特征
    select_k_model(forest, X_train, feat_labels, indices, importances)


if __name__ == '__main__':
    df_wine = pd.read_csv("../data/wine.data")
    df_wine.columns = ["Class label", "Alcohol", "Malic acid", "Ash",
                       "Alcalinity of ash", "Magnesium",
                       "Total phenols", "Flavanoids",
                       "Nonflavanoid phenols",
                       "Proanthocyanins", "Color intensity", "Hue",
                       "OD280/OD315 of diluted wines", "Proline"]
    print("Class Label:", np.unique(df_wine["Class label"]))

    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    # lbfgs 优化算法不支持l1正则化损失优化
    # 可以通过调整C值 来避免模型过拟合和欠拟合
    lr = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', multi_class='ovr')
    X_train_std, X_test_std = feature_scale("std", X_train, X_test)
    lr.fit(X_train_std, y_train)
    print("Train accuracy: ", lr.score(X_train_std, y_train))
    print("Test accuracy: ", lr.score(X_test_std, y_test))
    # lr.coef_ 对应的模型权重值

    plot_weight_cart(X_train_std, df_wine, y_train)

    select_k_feature(df_wine, X_train_std, y_train)
