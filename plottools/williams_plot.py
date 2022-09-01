# -*- coding: utf-8 -*-
# @Time    : 2022/4/27 04:48
# @Author  : FAN FAN
# @Site    : 
# @File    : williams_plot.py
# @Software: PyCharm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from data.scaler import Standardization

def make_inverse_matrix(X):
    X = np.column_stack((np.ones(X.shape[0]), X))
    influence_matrix = X.T.dot(X) + np.eye(X.shape[1]).dot(1e-9)
    return np.linalg.inv(influence_matrix)

def cal_leverage(i, X):
    '''
    :param x_i: descriptor row-vector of the query chemical
    :param X: descriptor matrix
    :param p: number of model variables
    :param n: number of training chemicals
    :return: hat_values: leverage of chemical
            warnings:
    '''
    inverse_matrix = make_inverse_matrix(X)
    X = np.column_stack((np.ones(X.shape[0]), X))
    hat_values = X[i, :].dot(inverse_matrix).dot(X[i, :])
    return hat_values


def draw(df_train, df_val, df_test, p, path):
    train_X = df_train.iloc[:, -p-1:-1].values
    val_X = df_val.iloc[:, -p-1:-1].values
    test_X = df_test.iloc[:, -p-1:-1].values
    #print(df_train.iloc[:, -p-1:-1])
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
    df_train.insert(loc=0, column='hat_value', value = h_train_list)
    df_val.insert(loc=0, column='hat_value', value = h_val_list)
    df_test.insert(loc=0, column='hat_value', value = h_test_list)
    df_train['subset'] = 'Train'
    df_val['subset'] = 'Validation'
    df_test['subset'] = 'Test'
    df = pd.concat([df_train, df_val, df_test], ignore_index=True)

    Scaling = Standardization(df['Residual'])
    df['Standardised_residuals'] = Scaling.Scaler(df['Residual'])
    df_train['Standardised_residuals'] = df['Standardised_residuals'].iloc[:len(df_train)].values
    df_val['Standardised_residuals'] = df['Standardised_residuals'].iloc[len(df_train):len(df_train) + len(df_val)].values
    df_test['Standardised_residuals'] = df['Standardised_residuals'].iloc[len(df_train) + len(df_val):].values

    g = sns.scatterplot(data=df, x='hat_value', y='Standardised_residuals', hue='subset', style='subset')
    #print(df_val['hat_value'][df_val['hat_value'] == df_val['hat_value'].max()].index)
    g.set(xlabel='Hat values', ylabel='Standardized Residuals')
    g.set(xlim=(0, 1.5), ylim=(-10, 20))
    plt.plot([warnings, warnings], [-100, 100], color='black', linestyle='dashed')
    plt.plot([0, warnings], [3, 3], color='black', linestyle='dashed')
    plt.plot([0, warnings], [-3, -3], color='black', linestyle='dashed')
    plt.savefig(path + 'op_william_plot.tiff', format='tiff', dpi=300)
    num_train_outlier = df_train['hat_value'][df_train['hat_value'] > warnings].index.tolist()
    num_val_outlier = df_val['hat_value'][df_val['hat_value'] > warnings].index.tolist()
    num_test_outlier = df_test['hat_value'][df_test['hat_value'] > warnings].index.tolist()
    num_train_resp_outlier = df_train['hat_value'][df_train['Standardised_residuals'] > 3].index.tolist()
    num_val_resp_outlier = df_val['hat_value'][df_val['Standardised_residuals'] > 3].index.tolist()
    num_test_resp_outlier = df_test['hat_value'][df_test['Standardised_residuals'] > 3].index.tolist()
    num_train_both_outlier = df_train['hat_value'][(df_train['hat_value'] > warnings)&(df_train['Standardised_residuals'] > 3)].index.tolist()
    num_val_both_outlier = df_val['hat_value'][(df_val['hat_value'] > warnings)&(df_val['Standardised_residuals'] > 3)].index.tolist()
    num_test_both_outlier = df_test['hat_value'][(df_test['hat_value'] > warnings)&(df_test['Standardised_residuals'] > 3)].index.tolist()

    print('Number of train set out AD(Structure/Response/Both):{}/{}/{}({}%)'.format(len(num_train_outlier), len(num_train_resp_outlier), 
                                                                                     len(num_train_both_outlier), 
                                                                                     round(100*(len(num_train_outlier) + len(num_train_resp_outlier) - len(num_train_both_outlier))/train_X.shape[0], 3)))
    #print('Number of train set out AD(Response):{}({}%)'.format(len(num_train_outlier), round(100 * len(num_train_outlier) / train_X.shape[0], 3)))
    print('Number of val set out AD(Structure/Response/Both):{}/{}/{}({}%)'.format(len(num_val_outlier), len(num_val_resp_outlier),
                                                                                     len(num_val_both_outlier),
                                                                                     round(100 * (len(num_val_outlier) + len(
                                                                                         num_val_resp_outlier) - len(num_val_both_outlier)) /
                                                                                           val_X.shape[0], 3)))
    #print('Number of val set out AD(Response):{}({}%)'.format(len(num_val_outlier), round(100 * len(num_val_outlier) / val_X.shape[0], 3)))
    print('Number of test set out AD(Structure/Response/Both):{}/{}/{}({}%)'.format(len(num_test_outlier), len(num_test_resp_outlier),
                                                                                     len(num_test_both_outlier),
                                                                                     round(100 * (len(num_test_outlier) + len(
                                                                                         num_test_resp_outlier) - len(num_test_both_outlier)) /
                                                                                           test_X.shape[0], 3)))
    #print('Number of test set out AD(Response):{}({}%)'.format(len(num_test_outlier), round(100 * len(num_test_outlier) / test_X.shape[0], 3)))
    
    df_outliers = pd.concat([df_train[(df_train['hat_value'] > warnings)|(abs(df_train['Standardised_residuals']) > 3)],
                             df_val[(df_val['hat_value'] > warnings)|(abs(df_val['Standardised_residuals']) > 3)],
                             df_test[(df_test['hat_value'] > warnings)|(abs(df_test['Standardised_residuals']) > 3)]], ignore_index=True)
    df_outliers['Outliers_Type'] = 0
    df_outliers.loc[df_outliers[df_outliers['hat_value'] > warnings].index.tolist(), 'Outliers_Type'] = 'Structure Outliers'
    df_outliers.loc[df_outliers[abs(df_outliers['Standardised_residuals']) > 3].index.tolist(), 'Outliers_Type'] = 'Response Outliers'
    df_outliers.loc[df_outliers[(df_outliers['hat_value'] > warnings)&(abs(df_outliers['Standardised_residuals']) > 3)].index.tolist(), 'Outliers_Type'] = 'Both'

    df_outliers.to_csv(path + 'op_william_plot_outliers.csv', index=False)


def draw_compare_3(df_train1, df_val1, df_test1, p1, df_train2, df_val2, df_test2, p2, df_train3, df_val3, df_test3, p3, path):
    train_X1 = df_train1.iloc[:, -p1-1:-1].values
    val_X1 = df_val1.iloc[:, -p1-1:-1].values
    test_X1 = df_test1.iloc[:, -p1-1:-1].values
    #print(df_train.iloc[:, -p-1:-1])
    X1 = np.concatenate((train_X1, val_X1, test_X1), axis=0)
    n1 = train_X1.shape[0]
    warnings1 = 3 * (p1 + 1) / n1
    h_list1 = []
    for i in range(X1.shape[0]):
        h_i = cal_leverage(i, X1)
        h_list1.append(h_i)

    h_train_list1 = h_list1[:train_X1.shape[0]]
    h_val_list1 = h_list1[train_X1.shape[0]:train_X1.shape[0]+val_X1.shape[0]]
    h_test_list1 = h_list1[train_X1.shape[0]+val_X1.shape[0]:]

    df1 = pd.concat([df_train1, df_val1, df_test1], axis=0)
    df1.insert(loc=0, column='hat_value', value=np.concatenate((h_train_list1, h_val_list1, h_test_list1), axis=0))
    df1['model'] = 'GCGAT'

    train_X2 = df_train2.iloc[:, -p2-1:-1].values
    val_X2 = df_val2.iloc[:, -p2-1:-1].values
    test_X2 = df_test2.iloc[:, -p2-1:-1].values
    #print(df_train.iloc[:, -p-1:-1])
    X2 = np.concatenate((train_X2, val_X2, test_X2), axis=0)
    n2 = train_X2.shape[0]
    warnings2 = 3 * (p2 + 1) / n1
    h_list2 = []
    for i in range(X2.shape[0]):
        h_i = cal_leverage(i, X2)
        h_list2.append(h_i)

    h_train_list2 = h_list2[:train_X2.shape[0]]
    h_val_list2 = h_list2[train_X2.shape[0]:train_X2.shape[0]+val_X2.shape[0]]
    h_test_list2 = h_list1[train_X2.shape[0]+val_X2.shape[0]:]

    #df2 = pd.DataFrame(np.concatenate((h_train_list2, h_val_list2, h_test_list2), axis=0), columns=['hat_value'])
    #df2['model'] = 'AGC'
    df2 = pd.concat([df_train2, df_val2, df_test2], axis=0)
    df2.insert(loc=0, column='hat_value', value=np.concatenate((h_train_list2, h_val_list2, h_test_list2), axis=0))
    df2['model'] = 'AGC'

    train_X3 = df_train3.iloc[:, -p3-1:-1].values
    val_X3 = df_val3.iloc[:, -p3-1:-1].values
    test_X3 = df_test3.iloc[:, -p3-1:-1].values
    #print(df_train.iloc[:, -p-1:-1])
    X3 = np.concatenate((train_X3, val_X3, test_X3), axis=0)
    n3 = train_X3.shape[0]
    warnings3 = 3 * (p3 + 1) / n1
    h_list3 = []
    for i in range(X3.shape[0]):
        h_i = cal_leverage(i, X3)
        h_list3.append(h_i)

    h_train_list3 = h_list3[:train_X3.shape[0]]
    h_val_list3 = h_list3[train_X3.shape[0]:train_X3.shape[0]+val_X3.shape[0]]
    h_test_list3 = h_list1[train_X3.shape[0]+val_X3.shape[0]:]

    #df3 = pd.DataFrame(np.concatenate((h_train_list3, h_val_list3, h_test_list3), axis=0), columns=['hat_value'])
    #df3['model'] = 'AFP'
    df3 = pd.concat([df_train3, df_val3, df_test3], axis=0)
    df3.insert(loc=0, column='hat_value', value=np.concatenate((h_train_list3, h_val_list3, h_test_list3), axis=0))
    df3['model'] = 'AFP'

    df = pd.concat([df1, df2, df3], ignore_index=True)
    Scaling = Standardization(df['Residual'])
    df['Standardised_residuals'] = Scaling.Scaler(df['Residual'])
    df1['Standardised_residuals'] = df['Standardised_residuals'].iloc[:741].values
    df2['Standardised_residuals'] = df['Standardised_residuals'].iloc[741:741*2].values
    df3['Standardised_residuals'] = df['Standardised_residuals'].iloc[741*2:741*3].values

    g = sns.scatterplot(data=df, x='hat_value', y='Standardised_residuals', hue='model', style='model', palette=['black', 'red', 'green'])
    g.set(xlabel='Hat values', ylabel='Standardized Residuals')
    g.set(xlim=(0, 1.5), ylim=(-10, 10))
    plt.plot([warnings1, warnings1], [-100, 100], color='black', linestyle='dashed')
    plt.plot([0, warnings1], [3, 3], color='black', linestyle='dashed')
    plt.plot([0, warnings1], [-3, -3], color='black', linestyle='dashed')

    plt.plot([warnings2, warnings2], [-100, 100], color='red', linestyle='dashed')
    plt.plot([0, warnings2], [3, 3], color='red', linestyle='dashed')
    plt.plot([0, warnings2], [-3, -3], color='red', linestyle='dashed')

    plt.plot([warnings3, warnings3], [-100, 100], color='green', linestyle='dashed')
    plt.plot([0, warnings3], [3, 3], color='green', linestyle='dashed')
    plt.plot([0, warnings3], [-3, -3], color='green', linestyle='dashed')

    plt.show()

    df_outliers = pd.concat([df1[(df1['hat_value'] > warnings1)|(abs(df1['Standardised_residuals']) > 3)],
                             df2[(df2['hat_value'] > warnings2)|(abs(df2['Standardised_residuals']) > 3)],
                             df3[(df3['hat_value'] > warnings3)|(abs(df3['Standardised_residuals']) > 3)]], ignore_index=True)

    df_outliers.to_csv(path + 'op_william_plot_outliers.csv', index=False)
