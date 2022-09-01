# -*- coding: utf-8 -*-
# @Time    : 2022/5/5 16:06
# @Author  : FAN FAN
# @Site    : 
# @File    : ensembling_frag.py
# @Software: PyCharm
from src.feature.atom_featurizer import classic_atom_featurizer
from src.feature.bond_featurizer import classic_bond_featurizer
from src.feature.mol_featurizer import classic_mol_featurizer
from utils.mol2graph import smiles_2_bigraph
from utils.junctiontree_encoder import JT_SubGraph

from utils.splitter import Splitter
from utils.metrics import Metrics
from utils.Earlystopping import EarlyStopping
from data.csv_dataset import MoleculeCSVDataset
from src.dgltools import collate_molgraphs, collate_fraggraphs
from data.dataloading import import_dataset
import torch
from torch.utils.data import DataLoader
import os
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.count_parameters import count_parameters

from networks.DMPNN import DMPNNNet
from networks.MPNN import MPNNNet
from networks.AttentiveFP import AttentiveFPNet
from networks.FraGAT import NewFraGATNet
from networks.AGC import AGCNet
from utils.piplines import train_epoch, evaluate, train_epoch_frag, evaluate_frag, PreFetch
from utils.Set_Seed_Reproducibility import set_seed

from data.model_library import save_model

params = {}
net_params = {}
#params['Dataset'] = 'HFUS'
params['init_lr'] = 10 ** -4.17
params['min_lr'] = 1e-9
params['weight_decay'] = 10 ** -6
params['lr_reduce_factor'] = 0.75
params['lr_schedule_patience'] = 30
params['earlystopping_patience'] = 100
params['max_epoch'] = 300

net_params['num_atom_type'] = 39
net_params['num_bond_type'] = 12
net_params['hidden_dim'] = 238
net_params['num_heads'] = 1
net_params['dropout'] = 0.034
net_params['depth'] = 3
net_params['layers'] = 1
net_params['residual'] = False
net_params['batch_norm'] = False
net_params['layer_norm'] = False
net_params['device'] = 'cuda'
dataset_list = ['ESOL']
splitting_seed = 2450


params['Dataset'] = dataset_list[0]
params['scheme'] = 'MG_plus_reference'
fragmentation = JT_SubGraph(scheme=params['scheme'])
df, scaling = import_dataset(params)
cache_file_path = os.path.realpath('./cache')
if not os.path.exists(cache_file_path):
    os.mkdir(cache_file_path)
cache_file = os.path.join(cache_file_path, params['Dataset'] + '_CCC')

error_path = os.path.realpath('./error_log')
if not os.path.exists(error_path):
    os.mkdir(error_path)
error_log_path = os.path.join(error_path, '{}_{}'.format(params['Dataset'], time.strftime('%Y-%m-%d-%H-%M')) + '.csv')

net_params['frag_dim'] = fragmentation.frag_dim
dataset = MoleculeCSVDataset(df, smiles_2_bigraph, classic_atom_featurizer, classic_bond_featurizer, classic_mol_featurizer, cache_file, load=True
                                 , error_log=error_log_path, fragmentation=fragmentation)

splitter = Splitter(dataset)

set_seed(seed=1000)
#random.seed(1000)
init_seed_list = [random.randint(0, 1000) for i in range(100)]

for i in range(100):
    torch.manual_seed(init_seed_list[i])
    train_set, val_set, test_set, raw_set = splitter.Random_Splitter(seed=splitting_seed, frac_train=0.8, frac_val=0.1)

    train_loader = DataLoader(train_set, collate_fn=collate_fraggraphs, batch_size=len(train_set), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))
    val_loader = DataLoader(val_set, collate_fn=collate_fraggraphs, batch_size=len(val_set), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))
    test_loader = DataLoader(test_set, collate_fn=collate_fraggraphs, batch_size=len(test_set), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))
    raw_loader = DataLoader(raw_set, collate_fn=collate_fraggraphs, batch_size=len(raw_set), shuffle=False, num_workers=0, worker_init_fn=np.random.seed(1000))

    fetched_data = PreFetch(train_loader, val_loader, test_loader, raw_loader, frag=True)
    model = NewFraGATNet(net_params).to(device='cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=params['lr_reduce_factor'],
                                                               patience=params['lr_schedule_patience'], verbose=False)
    t0 = time.time()
    per_epoch_time = []
    early_stopping = EarlyStopping(patience=params['earlystopping_patience'], path='checkpoint_ensemble' + params['Dataset'] + 'FraGAT' + '.pt')

    with tqdm(range(params['max_epoch'])) as t:
        n_param = count_parameters(model)
        for epoch in t:
            t.set_description('Epoch %d' % epoch)
            start = time.time()
            model, epoch_train_loss, epoch_train_metrics = train_epoch_frag(model, optimizer, scaling,
                                                                                fetched_data.train_iter, fetched_data.train_batched_origin_graph_list,
                                                                                fetched_data.train_batched_frag_graph_list,
                                                                                fetched_data.train_batched_motif_graph_list,
                                                                                fetched_data.train_targets_list,
                                                                                fetched_data.train_smiles_list,
                                                                                fetched_data.train_names_list, n_param)
            epoch_val_loss, epoch_val_metrics = evaluate_frag(model, scaling, fetched_data.val_iter, fetched_data.val_batched_origin_graph_list,
                                                                  fetched_data.val_batched_frag_graph_list, fetched_data.val_batched_motif_graph_list,
                                                                  fetched_data.val_targets_list, fetched_data.val_smiles_list, fetched_data.val_names_list, n_param)
            epoch_test_loss, epoch_test_metrics = evaluate_frag(model, scaling, fetched_data.test_iter, fetched_data.test_batched_origin_graph_list,
                                                                    fetched_data.test_batched_frag_graph_list, fetched_data.test_batched_motif_graph_list,
                                                                    fetched_data.test_targets_list, fetched_data.test_smiles_list, fetched_data.test_names_list, n_param)

            t.set_postfix({'time': time.time() - start, 'lr': optimizer.param_groups[0]['lr'],
                               'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss, 'test_loss': epoch_test_loss,
                               'train_R2': epoch_train_metrics.R2, 'val_R2': epoch_val_metrics.R2, 'test_R2': epoch_test_metrics.R2})
            per_epoch_time.append(time.time() - start)

            scheduler.step(epoch_val_loss)
            if optimizer.param_groups[0]['lr'] < params['min_lr']:
                print('\n! LR equal to min LR set.')
                break

            early_stopping(epoch_val_loss, model)
            if early_stopping.early_stop:
                break
    model = early_stopping.load_checkpoint(model)
    _, epoch_train_metrics = evaluate_frag(model, scaling, fetched_data.train_iter, fetched_data.train_batched_origin_graph_list,
                                             fetched_data.train_batched_frag_graph_list, fetched_data.train_batched_motif_graph_list,
                                             fetched_data.train_targets_list, fetched_data.train_smiles_list, fetched_data.train_names_list, n_param)
    _, epoch_val_metrics = evaluate_frag(model, scaling, fetched_data.val_iter, fetched_data.val_batched_origin_graph_list,
                                             fetched_data.val_batched_frag_graph_list, fetched_data.val_batched_motif_graph_list,
                                             fetched_data.val_targets_list, fetched_data.val_smiles_list, fetched_data.val_names_list, n_param)
    _, epoch_test_metrics = evaluate_frag(model, scaling, fetched_data.test_iter, fetched_data.test_batched_origin_graph_list,
                                             fetched_data.test_batched_frag_graph_list, fetched_data.test_batched_motif_graph_list,
                                             fetched_data.test_targets_list, fetched_data.test_smiles_list, fetched_data.test_names_list, n_param)
    _, epoch_raw_metrics = evaluate_frag(model, scaling, fetched_data.all_iter, fetched_data.all_batched_origin_graph_list,
                                             fetched_data.all_batched_frag_graph_list, fetched_data.all_batched_motif_graph_list,
                                             fetched_data.all_targets_list, fetched_data.all_smiles_list, fetched_data.all_names_list, n_param)


    path = '/Ensembles/HFUS_FraGAT_11'
    name = '{}_{}_{}'.format('Ensemble_0', params['Dataset'], 'FraGAT')
    results = pd.Series({'init_seed': init_seed_list[i], 'seed': splitting_seed, 'train_R2': epoch_train_metrics.R2, 'val_R2': epoch_val_metrics.R2,
                         'test_R2': epoch_test_metrics.R2, 'all_R2': epoch_raw_metrics.R2,
                         'train_MAE': epoch_train_metrics.MAE, 'val_MAE': epoch_val_metrics.MAE,
                         'test_MAE': epoch_test_metrics.MAE, 'all_MAE': epoch_raw_metrics.MAE,
                         'train_RMSE': epoch_train_metrics.RMSE, 'val_RMSE': epoch_val_metrics.RMSE,
                         'test_RMSE': epoch_test_metrics.RMSE, 'all_RMSE': epoch_raw_metrics.RMSE, 'train_SSE': epoch_train_metrics.SSE,
                         'val_SSE': epoch_val_metrics.SSE, 'test_SSE': epoch_test_metrics.SSE, 'all_SSE': epoch_raw_metrics.SSE,
                         'train_MAPE': epoch_train_metrics.MAPE, 'val_MAPE': epoch_val_metrics.MAPE, 'test_MAPE': epoch_test_metrics.MAPE,
                         'all_MAPE': epoch_raw_metrics.MAPE})
    comments = ''
    save_model(model, path, name, params, net_params, results, comments)




