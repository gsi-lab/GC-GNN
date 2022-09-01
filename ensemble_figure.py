# -*- coding: utf-8 -*-
# @Time    : 2022/5/6 02:25
# @Author  : FAN FAN
# @Site    : 
# @File    : ensemble_figure.py
# @Software: PyCharm
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os

model = 'AGC'
dataset = 'BMP'
index = 0
path = 'library/Ensembles/' + dataset + '_' + model + '_' + str(index) + '/'
df = pd.read_csv(path + 'Ensemble_0_' + dataset + '_' + model + '_settings.csv', sep=',', encoding='windows=1250', index_col=-1)


plot_list = ['MAE', 'RMSE', 'R2', 'SSE', 'MAPE']
labels_x = ['training', 'validation', 'testing']
labels_y = ['Mean Absolute Error (MAE) (-)', 'Root Mean Squared Error (RMSE) (-)', 'Coefficient of determination (R$^2$) (-)',
             'Sum of Squared Errors (SSE) (-)', 'Mean Absolute Percentage Error (MAPE) (%)']

results = {}
for i in range(len(labels_y)):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.boxplot([df['train_' + plot_list[i]], df['val_' + plot_list[i]], df['test_' + plot_list[i]]],
               vert=True, flierprops=dict(marker='+'), whiskerprops=dict(linestyle='--'), labels=labels_x)
    matplotlib.rcParams['axes.linewidth'] = 2
    ax.set_ylabel(labels_y[i])
    fig.tight_layout()
    fig.savefig(path + plot_list[i] + '.tiff', dpi=300, format='tiff')

    results['mean_train_' + plot_list[i]] = np.mean(df['train_' + plot_list[i]])
    results['mean_val_' + plot_list[i]] = np.mean(df['val_' + plot_list[i]])
    results['mean_test_' + plot_list[i]] = np.mean(df['test_' + plot_list[i]])
    results['mean_all_' + plot_list[i]] = np.mean(df['all_' + plot_list[i]])

    results['sd_train_' + plot_list[i]] = np.std(df['train_' + plot_list[i]])
    results['sd_val_' + plot_list[i]] = np.std(df['val_' + plot_list[i]])
    results['sd_test_' + plot_list[i]] = np.std(df['test_' + plot_list[i]])
    results['sd_all_' + plot_list[i]] = np.std(df['all_' + plot_list[i]])
    
    results['sem_train_' + plot_list[i]] = np.std(df['train_' + plot_list[i]], ddof=1) / np.sqrt(np.size(df['train_' + plot_list[i]]))
    results['sem_val_' + plot_list[i]] = np.std(df['val_' + plot_list[i]], ddof=1) / np.sqrt(np.size(df['val_' + plot_list[i]]))
    results['sem_test_' + plot_list[i]] = np.std(df['test_' + plot_list[i]], ddof=1) / np.sqrt(np.size(df['test_' + plot_list[i]]))
    results['sem_all_' + plot_list[i]] = np.std(df['all_' + plot_list[i]], ddof=1) / np.sqrt(np.size(df['all_' + plot_list[i]]))

df_summary = pd.DataFrame.from_dict(results, orient='index').T
df_summary.to_csv(path + 'Metrics_Summary_2.csv', index=False)

