# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/2/21 11:21
# Description:
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

df = pd.read_csv("../data/movie_data.csv", encoding='utf-8')
df = df.rename(columns={'0': 'review', '1': 'sentiment'})
# 最大文档率设置为10%， 以此排除文档中频繁出现的单词，因为这些频繁出现的单词可能是所有文档中都存在的常见词，不太可能与给定文档的特定主题相关
# 将单词限制为出现频率最高的5000个单词
count = CountVectorizer(stop_words='english', max_df=.1, max_features=5000)
# 得到词袋模型
X = count.fit_transform(df['review'].values)

# n_components 从文档中推断出10个不同的主题
# learning_method='batch' 可以让LDA估计器在以此迭代中使用所有可能的训练数据进行估计
lda = LatentDirichletAllocation(n_components=10, random_state=123, learning_method='batch')
X_topics = lda.fit_transform(X)

n_top_words = 5
# 词典表 索引对应对应的单词
feature_names = count.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    print(f'Topic {(topic_idx + 1)}')
    # print(topic.argsort()) 排序后的单词索引信息 取出后5个就是信息最多的索引 返回 feature_names[i] 找到对应的单词
    print(' '.join(feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]))
