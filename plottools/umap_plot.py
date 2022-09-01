# -*- coding: utf-8 -*-
# @Time    : 2022/8/8 15:16
# @Author  : FAN FAN
# @Site    : 
# @File    : umap_plot.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap


def draw_compare(df_train1, df_train2, df_val1, df_val2, df_test1, df_test2, path):
    df_train1.sort_values(by='Residual', inplace=True, ascending=True)
    df_val1.sort_values(by='Residual', inplace=True, ascending=True)
    df_test1.sort_values(by='Residual', inplace=True, ascending=True)

    train_X1 = df_train1.iloc[:, :-1].to_numpy()
    val_X1 = df_val1.iloc[:, :-1].to_numpy()
    test_X1 = df_test1.iloc[:, :-1].to_numpy()

    df1 = pd.concat([df_train1, df_val1, df_test1], axis=0)
    c_max1 = max(df1.iloc[:, -1])
    c_min1 = min(df1.iloc[:, -1])

    reducer = umap.UMAP(n_neighbors=15, min_dist=0, n_components=2, random_state=400)

    embedding_train1 = reducer.fit_transform(train_X1)
    embedding_val1 = reducer.transform(val_X1)
    embedding_test1 = reducer.transform(test_X1)

    embedding1 = np.concatenate((embedding_train1, embedding_val1, embedding_test1), axis=0)

    df_train2.sort_values(by='Residual', inplace=True, ascending=True)
    df_val2.sort_values(by='Residual', inplace=True, ascending=True)
    df_test2.sort_values(by='Residual', inplace=True, ascending=True)

    train_X2 = df_train2.iloc[:, :-1].to_numpy()
    val_X2 = df_val2.iloc[:, :-1].to_numpy()
    test_X2 = df_test2.iloc[:, :-1].to_numpy()

    df2 = pd.concat([df_train2, df_val2, df_test2], axis=0)
    c_max2 = max(df2.iloc[:, -1])
    c_min2 = min(df2.iloc[:, -1])

    reducer = umap.UMAP(n_neighbors=15, min_dist=0, n_components=2, random_state=400)

    embedding_train2 = reducer.fit_transform(train_X2)
    embedding_val2 = reducer.transform(val_X2)
    embedding_test2= reducer.transform(test_X2)

    embedding2 = np.concatenate((embedding_train2, embedding_val2, embedding_test2), axis=0)

    if c_max1 > c_max2:
        c_max = c_max1
    else:
        c_max = c_max2

    if c_min1 < c_min2:
        c_min = c_min1
    else:
        c_min = c_min2

    s1 = plt.scatter(embedding1[:, 0], embedding1[:, 1], edgecolors='None', c='r', marker='o', )
    s2 = plt.scatter(embedding2[:, 0], embedding2[:, 1], edgecolors='None', c='b', marker='s')

    plt.legend((s1, s2), ('HFOR', 'HFUS'), loc='best')

    #clb = plt.colorbar(orientation='vertical')
    #clb.ax.set_title('Residual')

    plt.show()

def draw(df_train, df_val, df_test, path):
    df_train.sort_values(by='Residual', inplace=True, ascending=True)
    df_val.sort_values(by='Residual', inplace=True, ascending=True)
    df_test.sort_values(by='Residual', inplace=True, ascending=True)

    train_X = df_train.iloc[:, :-1].to_numpy()
    val_X = df_val.iloc[:, :-1].to_numpy()
    test_X = df_test.iloc[:, :-1].to_numpy()

    df = pd.concat([df_train, df_val, df_test], axis=0)
    c_max = max(df.iloc[:, -1])
    c_min = min(df.iloc[:, -1])
    c_min = 0
    c_max = 10
    reducer = umap.UMAP(n_neighbors=15, min_dist=0, n_components=2, random_state=400)

    #object = StandardScaler()
    #train_X = object.fit_transform(train_X)
    #val_X = object.transform(val_X)
    #test_X = object.transform(test_X)

    #tsne = TSNE(perplexity=30, metric='euclidean', n_jobs=8, random_state=1000, verbose=True)

    embedding_train = reducer.fit_transform(train_X)
    embedding_val = reducer.transform(val_X)
    embedding_test = reducer.transform(test_X)

    #df_embedding_train = pd.DataFrame(data=embedding_train)
    #df_embedding_val = pd.DataFrame(data=embedding_val)
    #df_embedding_test = pd.DataFrame(data=embedding_test)
    #df_embedding_train['subset'] = 'Train'
    #df_embedding_val['subset'] = 'Validation'
    #df_embedding_test['subset'] = 'Test'
    #df_embedding = pd.concat([df_embedding_train, df_embedding_val, df_embedding_test], ignore_index=True)
    #g = sns.scatterplot(data=df_embedding, x=0, y=1, hue='subset', style='subset')
    fig = plt.figure()
    s1 = plt.scatter(embedding_train[:, 0], embedding_train[:, 1], edgecolors='None', c=df_train.iloc[:, -1].to_numpy(), vmin=c_min, vmax=c_max, cmap='bwr', marker='o', )
    s2 = plt.scatter(embedding_val[:, 0], embedding_val[:, 1], edgecolors='None', c=df_val.iloc[:, -1].to_numpy(), vmin=c_min, vmax=c_max, cmap='bwr', marker='s')
    s3 = plt.scatter(embedding_test[:, 0], embedding_test[:, 1], edgecolors='None', c=df_test.iloc[:, -1].to_numpy(), vmin=c_min, vmax=c_max, cmap='bwr', marker='^')
    plt.legend((s1, s2, s3), ('Train', 'Val', 'Test'), loc='best')
    plt.xlabel('UMAP-1')
    plt.ylabel('UMAP-2')
    #clb = plt.colorbar(orientation='vertical')
    #clb.ax.set_title('Residual')
    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #s1 = ax.scatter(embedding_train[:, 0], embedding_train[:, 1], embedding_train[:, 2], edgecolors='None', c=df_train.iloc[:, -1].to_numpy(), vmin=c_min, vmax=c_max, cmap='bwr', marker='o')
    #s2 = ax.scatter(embedding_val[:, 0], embedding_val[:, 1], embedding_val[:, 2], edgecolors='None', c=df_val.iloc[:, -1].to_numpy(), vmin=c_min, vmax=c_max, cmap='bwr', marker='s')
    #s3 = ax.scatter(embedding_test[:, 0], embedding_test[:, 1], embedding_test[:, 2], edgecolors='None', c=df_test.iloc[:, -1].to_numpy(), vmin=c_min, vmax=c_max, cmap='bwr', marker='^')
    #ax.legend((s1, s2, s3), ('Train', 'Val', 'Test'), loc='best')
    #clb = fig.colorbar(orientation='vertical')

    plt.show()
    fig.savefig(path + 'op_umap_plot.tiff', format='tiff', dpi=300)