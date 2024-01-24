from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from common.function import plot_decision_regions
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# 指定stratify可以使用np.bincount进行查询各个标签的分类出现的次数
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

print("Label count in y:", np.bincount(y))
print("Label count in y_train:", np.bincount(y_train))
print("Label count in y_test:", np.bincount(y_test))

# 进行特征缩放以获取最佳性能
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# 训练模型
ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print("Misclassified examples: %d" % (y_test != y_pred).sum())
print("Accuracy: %.3f" % accuracy_score(y_test, y_pred))
# or 结合predict
print("Accuracy: %.3f" % ppn.score(X_test_std, y_test))


# 展示决策树
def decision_tree(model_type, X_train, X_test, y_train, y_test):
    models = {
        "perceptron": Perceptron(eta0=0.1, random_state=1),
        "logistic": LogisticRegression(C=100, solver='lbfgs', multi_class='ovr'),
        "decision_tree": DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1),
        "random_forest": RandomForestClassifier(n_estimators=25, random_state=1, n_jobs=2),
        "neighbor": KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski'),
    }

    model = models.get(model_type)
    model.fit(X_train, y_train)
    X_combine = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X_combine, y_combined, classifier=model, test_idx=range(105, 150))
    plt.xlabel("Sepal length(cm)")
    plt.ylabel("Petal length(cm)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
