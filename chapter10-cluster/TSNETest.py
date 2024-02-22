# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/2/22 18:44
# Description:
from sklearn.manifold import TSNE
import numpy as np

# 假设X_high_dim是你的高维数据集，其形状为(n_samples, n_features)
X_high_dim = np.random.rand(100, 50)  # 这只是一个示例，你应该用你的实际数据替换它

# 创建一个t-SNE对象
tsne = TSNE(n_components=2, random_state=0)

# 使用t-SNE进行降维
X_low_dim = tsne.fit_transform(X_high_dim)
# X_low_dim现在就是你的二维数据集，其形状为(n_samples, 2)
# 可以使用降维后的数据进行聚类分析
