# -*- coding: utf-8 -*-
# @Time    : 2022/5/9 10:26
# @Author  : FAN FAN
# @Site    : 
# @File    : ensemble_frag_compound.py
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
from utils.piplines import evaluate, PreFetch, evaluate_descriptors
from utils.Set_Seed_Reproducibility import set_seed

from data.model_library import save_model, load_model, load_optimal_model

params = {}
net_params = {}
#params['Dataset'] = 'HFUS'
dataset_list = ['ESOL']
splitting_seed = [3348]
path_l = ['Ensembles/ESOL_AFP_0']
name_l = ['Ensemble_0_ESOL_AFP']

for j in range(len(splitting_seed)):
    params['Dataset'] = dataset_list[j]
    df, scaling = import_dataset(params)
    cache_file_path = os.path.realpath('./cache')
    if not os.path.exists(cache_file_path):
        os.mkdir(cache_file_path)
    cache_file = os.path.join(cache_file_path, params['Dataset'] + '_CCC')

    error_path = os.path.realpath('./error_log')
    if not os.path.exists(error_path):
        os.mkdir(error_path)
    error_log_path = os.path.join(error_path, '{}_{}'.format(params['Dataset'], time.strftime('%Y-%m-%d-%H-%M')) + '.csv')

    dataset = MoleculeCSVDataset(df, smiles_2_bigraph, classic_atom_featurizer, classic_bond_featurizer, classic_mol_featurizer, cache_file, load=True
                                     , error_log=error_log_path)

    splitter = Splitter(dataset)

    set_seed(seed=1000)
    train_set, val_set, test_set, raw_set = splitter.Random_Splitter(seed=splitting_seed[j], frac_train=0.8, frac_val=0.1)

    train_loader = DataLoader(train_set, collate_fn=collate_molgraphs, batch_size=len(train_set), shuffle=False, num_workers=0,
                              worker_init_fn=np.random.seed(1000))
    val_loader = DataLoader(val_set, collate_fn=collate_molgraphs, batch_size=len(val_set), shuffle=False, num_workers=0,
                            worker_init_fn=np.random.seed(1000))
    test_loader = DataLoader(test_set, collate_fn=collate_molgraphs, batch_size=len(test_set), shuffle=False, num_workers=0,
                             worker_init_fn=np.random.seed(1000))
    raw_loader = DataLoader(raw_set, collate_fn=collate_molgraphs, batch_size=len(raw_set), shuffle=False, num_workers=0,
                            worker_init_fn=np.random.seed(1000))

    fetched_data = PreFetch(train_loader, val_loader, test_loader, raw_loader, frag=False)

    path = path_l[j]
    name = name_l[j]

    for i in range(100):
        name_idx = name + '_' + str(i)
        params, net_params, model = load_model(path, name_idx)
        n_param = count_parameters(model)
        _, _, train_predict, train_target, train_smiles = evaluate(model, scaling, fetched_data.train_iter, fetched_data.train_batched_origin_graph_list,
                                             fetched_data.train_targets_list, fetched_data.train_smiles_list, fetched_data.train_names_list, n_param, flag=True)
        _, _, val_predict, val_target, val_smiles = evaluate(model, scaling, fetched_data.val_iter, fetched_data.val_batched_origin_graph_list,
                                             fetched_data.val_targets_list, fetched_data.val_smiles_list, fetched_data.val_names_list, n_param, flag=True)
        _, _, test_predict, test_target, test_smiles = evaluate(model, scaling, fetched_data.test_iter, fetched_data.test_batched_origin_graph_list,
                                             fetched_data.test_targets_list, fetched_data.test_smiles_list, fetched_data.test_names_list, n_param, flag=True)
        if i == 0:
            df_train = pd.DataFrame({'SMILES': train_smiles[0], 'Tag': 'Train', 'Target': train_target.numpy().flatten().tolist(),
                                     'Predict_'+str(i): train_predict.numpy().flatten().tolist()})
            df_val = pd.DataFrame({'SMILES': val_smiles[0], 'Tag': 'Val', 'Target': val_target.numpy().flatten().tolist(),
                                     'Predict_'+str(i): val_predict.numpy().flatten().tolist()})
            df_test = pd.DataFrame({'SMILES': test_smiles[0], 'Tag': 'Test', 'Target': test_target.numpy().flatten().tolist(),
                                     'Predict_'+str(i): test_predict.numpy().flatten().tolist()})
        else:
            df_train['Predict_'+str(i)] = train_predict.numpy().flatten().tolist()
            df_val['Predict_' + str(i)] = val_predict.numpy().flatten().tolist()
            df_test['Predict_' + str(i)] = test_predict.numpy().flatten().tolist()

    df_results = pd.concat([df_train, df_val, df_test], axis=0, ignore_index=True, sort=False)

    op_idx, init_seed, seed, params, net_params, model = load_optimal_model(path, name)
    all_smiles, all_descriptors = evaluate_descriptors(model, scaling, fetched_data.all_iter, fetched_data.all_batched_origin_graph_list,
                                                                    fetched_data.all_targets_list, fetched_data.all_smiles_list,
                                                                    fetched_data.all_names_list, n_param)

    df_descriptors = pd.DataFrame(all_descriptors.detach().to(device='cpu').numpy())
    df_descriptors['SMILES'] = all_smiles[0]

    df_merge = pd.merge(df_results, df_descriptors, how='outer', on='SMILES')

    save_file_path = os.path.join('./library/' + path, '{}_{}_{}_{}_{}'.format(name, 'compound_descriptors', seed, 'OP'+str(op_idx), time.strftime('%Y-%m-%d-%H-%M')) + '.csv')
    df_merge.to_csv(save_file_path, index=False)
