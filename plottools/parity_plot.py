# -*- coding: utf-8 -*-
# @Time    : 2022/6/8 14:03
# @Author  : FAN FAN
# @Site    : 
# @File    : parity_plot.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np


def parity_plot(df_train, df_val, df_test, path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(df_train['Target'], df_train['Predict'], 'bo', label='train')
    ax.plot(df_val['Target'], df_val['Predict'], 'ro', label='val')
    ax.plot(df_test['Target'], df_test['Predict'], 'co', label='test')
    ax.plot(np.concatenate((df_train['Target'], df_val['Target'], df_test['Target'])),
            np.concatenate((df_train['Target'], df_val['Target'], df_test['Target'])), 'k-')

    plt.legend(loc='upper left')

    fig.savefig(path + 'parity_plot.tiff', bbox_inches='tight', format='tiff', dpi=300)