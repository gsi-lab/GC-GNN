# -*- coding: utf-8 -*-
# @Time    : 2022/4/14 16:05
# @Author  : FAN FAN
# @Site    : 
# @File    : model_library.py
# @Software: PyCharm
import os
import torch
import pandas as pd
import time
import re

from networks.DMPNN import DMPNNNet as DMPNN
from networks.MPNN import MPNNNet as MPNN
from networks.AttentiveFP import AttentiveFPNet as AFP
from networks.FraGAT import NewFraGATNet as FraGAT
from networks.AGC import AGCNet as AGC

def save_model(model, path, name, params, net_params, results, comment):
    lib_file_path = os.path.realpath('.//library/' + path)
    if not os.path.exists(lib_file_path):
        os.makedirs(lib_file_path)
    model_file_path = os.path.join(lib_file_path, name + '_0.pt')
    def rename_file(filename):
        i = 1
        def check_meta(i, file_name):
            new_file_name = file_name
            if os.path.isfile(file_name):
                new_file_name = re.sub('_(\d*).pt', '_{}.pt'.format(i), file_name)
                i += 1
            if os.path.isfile(new_file_name):
                i, new_file_name = check_meta(i, file_name)
            return i, new_file_name
        return check_meta(i, filename)
    i, model_file_path = rename_file(model_file_path)
    torch.save(model.state_dict(), model_file_path)
    df1 = pd.DataFrame.from_dict(params, orient='index').T
    df2 = pd.DataFrame.from_dict(net_params, orient='index').T
    df3 = results.to_frame().T
    for _, param in enumerate(df1.columns):
        df1.rename(columns={param : 'param:'+param}, inplace=True)
    for _, net_param in enumerate(df2.columns):
        df2.rename(columns={net_param : 'net_param:'+net_param}, inplace=True)
    df_merge = pd.concat([df1, df2, df3], axis=1)
    df_merge['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    df_merge['comment'] = comment
    df_merge['index'] = i-1
    setting_file_path = os.path.join(lib_file_path, name + '_settings' + '.csv')
    if os.path.exists(setting_file_path):
        df_merge.to_csv(setting_file_path, mode='a', header=False, index=False)
    else:
        df_merge.to_csv(setting_file_path, mode='w', index=False)


def load_model(path, name_idx):
    lib_file_path = os.path.realpath('.//library/' + path)
    model_file_path = os.path.join(lib_file_path, name_idx + '.pt')

    directory, filename = os.path.split(model_file_path)
    idx = int(re.findall('_(\d+).pt', filename)[0])
    name = re.findall('(\w*)_\d*.pt', filename)[0]

    setting_file_path = os.path.join(lib_file_path, name + '_settings' + '.csv')
    df = pd.read_csv(setting_file_path, sep=',', encoding='windows=1250', index_col=-1)
    params = {}
    net_params = {}
    for _, item in enumerate(df.columns):
        if ':' in item:
            if item.split(':')[0] == 'param':
                params[item.split(':')[1]] = df[item][idx]
            if item.split(':')[0] == 'net_param':
                net_params[item.split(':')[1]] = df[item][idx]

    network = re.findall('\w*_(\w*)_\d*.pt', filename)[0]
    model = eval(network)(net_params).to(device='cuda')

    model.load_state_dict(torch.load(model_file_path), strict=False)
    return params, net_params, model

def load_params(name, idx):
    setting_file_path = os.path.join(name + '.csv')
    df = pd.read_csv(setting_file_path, sep=',', encoding='windows=1250', index_col=-1)
    params = {}
    net_params = {}
    for _, item in enumerate(df.columns):
        if ':' in item:
            if item.split(':')[0] == 'param':
                params[item.split(':')[1]] = df[item][idx]
            if item.split(':')[0] == 'net_param':
                net_params[item.split(':')[1]] = df[item][idx]
    params['init_seed'] = df['init_seed'][idx]
    return params, net_params

def load_optimal_model(path, name):
    lib_file_path = os.path.realpath('.//library/' + path)

    setting_file_path = os.path.join(lib_file_path, name + '_settings' + '.csv')
    df = pd.read_csv(setting_file_path, sep=',', encoding='windows=1250', index_col=-1)

    idx = df[['test_RMSE']].idxmin()['test_RMSE']
    name_idx = name + '_' + str(idx)
    model_file_path = os.path.join(lib_file_path, name_idx + '.pt')
    params = {}
    net_params = {}

    for _, item in enumerate(df.columns):
        if ':' in item:
            if item.split(':')[0] == 'param':
                params[item.split(':')[1]] = df[item][idx]
            if item.split(':')[0] == 'net_param':
                net_params[item.split(':')[1]] = df[item][idx]

    init_seed = df['init_seed'][idx]
    seed = df['seed'][idx]
    network = re.findall('\w*_(\w*)_\d*', name_idx)[0]
    model = eval(network)(net_params).to(device='cuda')
    model.load_state_dict(torch.load(model_file_path), strict=False)
    return idx, init_seed, seed, params, net_params, model



"""
def test_save_param(params, net_params, comment):
    if not os.path.exists('.//library'):
        os.mkdir('.//library')
    setting_file_path = os.path.join('.//library', 'test' + '_settings' + '.csv')
    df1 = pd.DataFrame.from_dict(params, orient='index').T
    df2 = pd.DataFrame.from_dict(net_params, orient='index').T
    for _, param in enumerate(df1.columns):
        df1.rename(columns={param : 'param:'+param}, inplace=True)
    for _, net_param in enumerate(df2.columns):
        df2.rename(columns={net_param : 'net_param:'+net_param}, inplace=True)
    df_merge = pd.concat([df1, df2], axis=1)
    df_merge['comment'] = comment
    if os.path.exists(setting_file_path):
        df_merge.to_csv(setting_file_path, mode='a', header=False)
    else:
        df_merge.to_csv(setting_file_path, mode='w')


def test_load_param(model_name):
    df = pd.read_csv('.//library/' + model_name + '_settings.csv', sep=',', encoding='windows=1250', index_col=-1)
    params = {}
    net_params = {}
    for _, item in enumerate(df.columns):
        if ':' in item:
            if item.split(':')[0] == 'param':
                params[item.split(':')[1]] = df[item][0]
            if item.split(':')[0] == 'net_param':
                net_params[item.split(':')[1]] = df[item][0]
"""