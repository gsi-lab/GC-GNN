# -*- coding: utf-8 -*-
# @Time    : 2022/6/8 14:22
# @Author  : FAN FAN
# @Site    : 
# @File    : confidence_intervals_plot.py
# @Software: PyCharm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import os
import numpy as np


def draw_conf_plot(df, label_x, label_y, path):
    U_breach = ((df['Target'] > df['UCI']).sum() / len(df) * 100).round(3)
    L_breach = ((df['Target'] < df['LCI']).sum() / len(df) * 100).round(3)
    coverage = (df['cover'].sum() / len(df) * 100).round(3)
    # Save to txt
    uncert_quant_to = os.path.join(path, 'overview.txt')
    uncert_quant = {'Coverage': coverage,
                    'Lower CI breach': L_breach,
                    'Upper CI breach': U_breach}
    print(uncert_quant, file=open(uncert_quant_to, 'a'))
    print("\n", file=open(uncert_quant_to, "a"))

    fig, ax = plt.subplots()
    tgt_sorted = df.sort_values('Target')
    x = np.arange(len(tgt_sorted))
    ax.plot(x, tgt_sorted['Target'], linestyle='', marker='.', color='black', label='target')
    ax.fill_between(x, tgt_sorted['LCI'], tgt_sorted['UCI'], alpha=0.5, color='red', label='95% CI')
    ax.set_ylabel(label_y)
    ax.set_xlabel('Compound index')
    anchored_text = AnchoredText(label_x, loc='lower right')
    ax.add_artist(anchored_text)
    plt.legend(loc='upper left');
    plt.show()
    fig.savefig(path + label_x + '_cov_plot.tiff', dpi=300, format='tiff')

