# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/2/20 15:57
# Description:
import re
from nltk.stem.porter import PorterStemmer
import pyprind
import pandas as pd
import os
import sys
import numpy as np
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


# 加载文件夹中的数据到csv文件中
def load_move_to_csv():
    base_path = '../data/aclImdb'
    labels = {'pos': 1, 'neg': 0}
    pbar = pyprind.ProgBar(50000, stream=sys.stdout)

    df = pd.DataFrame()
    for s in ('test', 'train'):
        for l in ('pos', 'neg'):
            path = os.path.join(base_path, s, l)
            for file in sorted(os.listdir(path)):
                with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                    txt = infile.read()
                df = df._append([[txt, labels[l]]], ignore_index=True)
                pbar.update()

    df.columns = ['review', 'sentiment']
    np.random.seed(0)
    df = df.reindex(np.random.permutation(df.index))
    df.to_csv('../data/movie_data.csv', index=False, encoding='utf-8')


df = pd.read_csv("../data/movie_data.csv", encoding='utf-8')


def preprocessor(text):
    # 删除所有的html标签
    text = re.sub("<[^>]*>", "", text)
    emoticons = re.findall("(?::|;|=)(?:-)?(?:\)|\(|D|P)", text)
    # 删除文本中所有非单词字符 并且大写转小写
    text = (re.sub("[\W]+", " ", text.lower()) + " ".join(emoticons).replace('-', ''))
    return text


# print(preprocessor(df.loc[0, 'review'][-50:]))
df['review'] = df['review'].apply(preprocessor)
porter = PorterStemmer()


# 将文档处理成token 词干提取法
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


def tokenizer(text):
    return text.split()


# print(tokenizer("is seven title brazil not available"))
stop = stopwords.words("english")
# res = [w for w in tokenizer_porter('a runner likes running and runs a lot') if w not in stop]
# print(res)
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
small_param_grid = [
    {
        'vect__ngram_range': [(1, 1)],
        'vect__stop_words': [None],
        'vect__tokenizer': [tokenizer_porter, tokenizer],
        'clf__penalty': ['l2'],
        'clf__C': [1.0, 10.0]
    },
    {
        'vect__ngram_range': [(1, 1)],
        'vect__stop_words': [(stop, None)],
        'vect__tokenizer': [tokenizer],
        'vect__use_idf': [False],
        'vect__norm': [None],
        'clf__penalty': ['l2'],
        'clf__C': [1.0, 10.0]
    }
]

lr_tfidf = Pipeline([
    ('vect', tfidf),
    ('clf', LogisticRegression(solver='liblinear'))
])

gs_lr_tfidf = GridSearchCV(lr_tfidf, small_param_grid, scoring='accuracy', cv=5, verbose=2, n_jobs=-1)
gs_lr_tfidf.fit(X_train, y_train)

# Best parameter set :
# {'clf__C': 10.0, 'clf__penalty': 'l2', 'vect__ngram_range': (1, 1), 'vect__stop_words': None,
# 'vect__tokenizer': <function tokenizer at 0x1272e74c0>}
print(f'Best parameter set : {gs_lr_tfidf.best_params_}')
print(f'CV Accuracy: {gs_lr_tfidf.best_score_:.3f}')
print(f'Test Accuracy: {gs_lr_tfidf.best_estimator_.score(X_test, y_test):.3f}')


