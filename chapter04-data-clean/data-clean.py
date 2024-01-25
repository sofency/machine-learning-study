import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.impute import SimpleImputer


def drop_nan(df: DataFrame):
    print("统计没列属性空值个数")
    print(df.isnull().sum())
    print("axis=0表示行 axis=1 表示列")
    print(df.dropna(axis=0))
    print("只删除所有行所有元素都是NAN的")
    print(df.dropna(how="all"))
    print("如果一行非空数据少于4个则删除")
    print(df.dropna(thresh=4))
    print("只删除某一列出现nan的行")
    print(df.dropna(subset=['C']))


def fill_nan_methods_sklearn(df: DataFrame):
    # sklearn方式
    # strategy = median(中位数) 或者most_frequent (众数)
    imr = SimpleImputer(missing_values=np.nan, strategy='mean')
    imr = imr.fit(df.values)
    imputed_data = imr.transform(df.values)
    print(imputed_data)


def fill_nan_methods_pandas(df: DataFrame):
    df.fillna(df.mean())


def onehot_sklearn(df: DataFrame):
    # sklearn 方式
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    X = df[["color", "size", "price"]].values
    # 只针对第一列进行独热编码
    c_trans = ColumnTransformer([
        ("onehot", OneHotEncoder(), [0]),
        ("nothing", 'passthrough', [1, 2])
    ])
    print(c_trans.fit_transform(X).astype(float))


def pandas_generate_dataframe():
    df = pd.DataFrame([
        ["green", 'M', 10.1, 'class2'],
        ["red", 'L', 13.5, 'class1'],
        ["blue", 'XL', 15.3, 'class2']
    ])
    df.columns = ['color', 'size', 'price', 'label']
    size_mapping = {"XL": 3, "L": 2, "M": 1}
    df["size"] = df["size"].map(size_mapping)
    # enumerate 返回索引
    # class_mapping = {label: idx for idx, label in enumerate(np.unique(df["label"]))}
    # 对于无分类标签 随意赋值数字即可
    # df["label"] = df["label"].map(class_mapping)
    # 或者
    from sklearn.preprocessing import LabelEncoder
    class_label = LabelEncoder()
    df["label"] = class_label.fit_transform(df["label"].values)
    # 转换为原来的标签
    # df["label"] = class_label.inverse_transform(df["label"])

    # color_label = LabelEncoder()
    # df["color"] = color_label.fit_transform(df["color"].values)
    # 转化后出现 green颜色大于blue 的情况，这种是不合理的， 因此采用独热编码
    # pandas 独热编码 只会转换DataFrame中的字符串列 其他保持不变
    df = pd.get_dummies(df[['color', 'size', 'price']])
    print(df)


if __name__ == '__main__':
    # df = pd.read_csv("../data/demo.csv")
    pandas_generate_dataframe()
