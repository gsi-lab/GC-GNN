# -*- coding: utf-8 -*-
# @Time    : 2022/7/18 13:52
# @Author  : FAN FAN
# @Site    : 
# @File    : t_sne_plot.py
# @Software: PyCharm
from openTSNE import TSNEEmbedding
from openTSNE import affinity
from openTSNE import initialization
from openTSNE import TSNE

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .williams_plot import cal_leverage

def draw(df_train, df_val, df_test, path):
    train_X = df_train.iloc[:, :-1].to_numpy()
    val_X = df_val.iloc[:, :-1].to_numpy()
    test_X = df_test.iloc[:, :-1].to_numpy()

    #object = StandardScaler()
    #train_X = object.fit_transform(train_X)
    #val_X = object.transform(val_X)
    #test_X = object.transform(test_X)

    tsne = TSNE(perplexity=30, metric='euclidean', n_jobs=8, random_state=1000, verbose=True)

    embedding_train = tsne.fit(train_X)

    embedding_val = embedding_train.transform(val_X)
    embedding_test = embedding_train.transform(test_X)

    df_embedding_train = pd.DataFrame(data=embedding_train)
    df_embedding_val = pd.DataFrame(data=embedding_val)
    df_embedding_test = pd.DataFrame(data=embedding_test)
    df_embedding_train['subset'] = 'Train'
    df_embedding_val['subset'] = 'Validation'
    df_embedding_test['subset'] = 'Test'
    df_embedding = pd.concat([df_embedding_train, df_embedding_val, df_embedding_test], ignore_index=True)
    g = sns.scatterplot(data=df_embedding, x=0, y=1, hue='subset', style='subset')
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.savefig(path + 'op_t_sne_plot.tiff', format='tiff', dpi=900)
    plt.close()

def draw_PCA(df_train, df_val, df_test, path):
    train_X = df_train.iloc[:, :-1].to_numpy()
    val_X = df_val.iloc[:, :-1].to_numpy()
    test_X = df_test.iloc[:, :-1].to_numpy()

    pca = PCA(n_components=2)
    embedding_train = pca.fit_transform(train_X)
    embedding_val = pca.transform(val_X)
    embedding_test = pca.transform(test_X)

    df_embedding_train = pd.DataFrame(data=embedding_train)
    df_embedding_val = pd.DataFrame(data=embedding_val)
    df_embedding_test = pd.DataFrame(data=embedding_test)
    df_embedding_train['subset'] = 'Train'
    df_embedding_val['subset'] = 'Validation'
    df_embedding_test['subset'] = 'Test'
    df_embedding = pd.concat([df_embedding_train, df_embedding_val, df_embedding_test], ignore_index=True)
    g = sns.scatterplot(data=df_embedding, x=0, y=1, hue='subset', style='subset')
    plt.xlabel('PCA Feature 1')
    plt.ylabel('PCA Feature 2')
    plt.savefig(path + 'op_PCA_plot.tiff', format='tiff', dpi=900)
    plt.close()

def draw_PCA_william(df_train, df_val, df_test, path):
    train_X = df_train.iloc[:, :-1].to_numpy()
    val_X = df_val.iloc[:, :-1].to_numpy()
    test_X = df_test.iloc[:, :-1].to_numpy()

    pca = PCA(n_components=2)
    embedding_train = pca.fit_transform(train_X)
    embedding_val = pca.transform(val_X)
    embedding_test = pca.transform(test_X)

    df_embedding_train = pd.DataFrame(data=embedding_train)
    df_embedding_val = pd.DataFrame(data=embedding_val)
    df_embedding_test = pd.DataFrame(data=embedding_test)
    df_embedding_train['subset'] = 'Train'
    df_embedding_val['subset'] = 'Validation'
    df_embedding_test['subset'] = 'Test'

    X = np.concatenate((train_X, val_X, test_X), axis=0)
    n = train_X.shape[0]
    p = 64
    warnings = 3 * (p + 1) / n
    h_list = []
    for i in range(X.shape[0]):
        h_i = cal_leverage(i, X)
        h_list.append(h_i)

    h_train_list = h_list[:train_X.shape[0]]
    h_val_list = h_list[train_X.shape[0]:train_X.shape[0]+val_X.shape[0]]
    h_test_list = h_list[train_X.shape[0]+val_X.shape[0]:]
    df_train.insert(loc=3, column='hat_value', value = h_train_list)
    df_val.insert(loc=3, column='hat_value', value = h_val_list)
    df_test.insert(loc=3, column='hat_value', value = h_test_list)
    df_train.reset_index(inplace=True)
    df_val.reset_index(inplace=True)
    df_test.reset_index(inplace=True)
    #num_train_outlier = df_train['hat_value'][df_train['hat_value'] > warnings].index.tolist()
    #num_val_outlier = df_val['hat_value'][df_val['hat_value'] > warnings].index.tolist()
    #num_test_outlier = df_test['hat_value'][df_test['hat_value'] > warnings].index.tolist()

    df_embedding_train.loc[df_train[df_train['hat_value'] > warnings].index.tolist(), 'subset'] = 'Train Outliers'
    df_embedding_val.loc[df_val[df_val['hat_value'] > warnings].index.tolist(), 'subset'] = 'Val Outliers'
    df_embedding_test.loc[df_test[df_test['hat_value'] > warnings].index.tolist(), 'subset'] = 'Test Outliers'

    df_embedding = pd.concat([df_embedding_train, df_embedding_val, df_embedding_test], ignore_index=True)
    g = sns.scatterplot(data=df_embedding, x=0, y=1, hue='subset', style='subset')
    plt.savefig(path + 'op_PCA_WilliamOutliers_plot.tiff', format='tiff', dpi=300)


def draw_bkp(df_train, df_val, df_test, path):
    train_X = df_train.iloc[:, :-1].to_numpy()
    val_X = df_val.iloc[:, :-1].to_numpy()
    test_X = df_test.iloc[:, :-1].to_numpy()

    #tsne = TSNE(perplexity=30, metric='euclidean', n_jobs=8, random_state=1000, verbose=True)
    affinities_train = affinity.PerplexityBasedNN(train_X, perplexity=30, metric='euclidean', n_jobs=8, random_state=1000, verbose=True)
    init_train = initialization.pca(train_X, random_state=1000)
    embedding_train = TSNEEmbedding(init_train, affinities_train, negative_gradient_method='fft', n_jobs=8, verbose=True)

    embedding_train_1 = embedding_train.optimize(n_iter=250, exaggeration=12, momentum=0.5)
    embedding_train_2 = embedding_train_1.optimize(n_iter=500, momentum=0.8)

    embedding_val = embedding_train_2.prepare_partial(val_X, initialization='median', k=25, perplexity=5, )
    embedding_val_1 = embedding_val.optimize(n_iter=250, learning_rate=0.1, momentum=0.8)

    embedding_test = embedding_train_2.prepare_partial(test_X, initialization='median', k=25, perplexity=5, )
    embedding_test_1 = embedding_test.optimize(n_iter=250, learning_rate=0.1, momentum=0.8)

    df_embedding_train = pd.DataFrame(data=embedding_train_2)
    df_embedding_val = pd.DataFrame(data=embedding_val_1)
    df_embedding_test = pd.DataFrame(data=embedding_test_1)
    df_embedding_train['subset'] = 'Train'
    df_embedding_val['subset'] = 'Validation'
    df_embedding_test['subset'] = 'Test'
    df_embedding = pd.concat([df_embedding_train, df_embedding_val, df_embedding_test], ignore_index=True)
    g = sns.scatterplot(data=df_embedding, x=0, y=1, hue='subset', style='subset')
    plt.savefig(path + 'op_t_sne_advanced_plot.tiff', format='tiff', dpi=300)
    '''
    utils.plot(embedding_train_1, train_Y, colors=utils.MACOSKO_COLORS)
    X = np.concatenate((train_X, val_X, test_X), axis=0)
    n = train_X.shape[0]
    warnings = 3 * (p + 1) / n
    h_list = []
    for i in range(X.shape[0]):
        h_i = cal_leverage(i, X)
        h_list.append(h_i)

    h_train_list = h_list[:train_X.shape[0]]
    h_val_list = h_list[train_X.shape[0]:train_X.shape[0]+val_X.shape[0]]
    h_test_list = h_list[train_X.shape[0]+val_X.shape[0]:]
    df_train.insert(loc=3, column='hat_value', value = h_train_list)
    df_val.insert(loc=3, column='hat_value', value = h_val_list)
    df_test.insert(loc=3, column='hat_value', value = h_test_list)
    df_train['subset'] = 'Train'
    df_val['subset'] = 'Validation'
    df_test['subset'] = 'Test'
    df = pd.concat([df_train, df_val, df_test], ignore_index=True)
    '''