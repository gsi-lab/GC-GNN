# -*- coding: utf-8 -*-
# @Time    : 2022/3/23 16:41
# @Author  : FAN FAN
# @Site    : 
# @File    : piplines.py
# @Software: PyCharm
import time

import dgl
import torch
from .metrics import Metrics
from .Set_Seed_Reproducibility import set_seed


class PreFetch(object):
    def __init__(self, train_loader, val_loader, test_loader, raw_loader, frag):
        if frag == 2:
            self.train_batched_origin_graph_list, self.train_batched_frag_graph_list, self.train_batched_motif_graph_list, \
            self.train_batched_channel_graph_list, self.train_batched_index_list, self.train_targets_list, self.train_smiles_list, \
            self.train_names_list, self.train_iter = [], [], [], [], [], [], [], [], []
            for iter, batch in enumerate(train_loader):
                _batched_origin_graph, _batched_frag_graph, _batched_motif_graph, _batched_channel_graph, _batched_index_list, _targets, _smiles, _names = batch
                self.train_batched_origin_graph_list.append(_batched_origin_graph)
                self.train_batched_frag_graph_list.append(_batched_frag_graph)
                self.train_batched_motif_graph_list.append(_batched_motif_graph)
                self.train_batched_channel_graph_list.append(_batched_channel_graph)
                self.train_batched_index_list.append(_batched_index_list)
                self.train_targets_list.append(_targets)
                self.train_smiles_list.append(_smiles)
                self.train_names_list.append(_smiles)
                self.train_iter.append(iter)

            self.val_batched_origin_graph_list, self.val_batched_frag_graph_list, self.val_batched_motif_graph_list, \
            self.val_batched_channel_graph_list, self.val_batched_index_list, self.val_targets_list, self.val_smiles_list, \
            self.val_names_list, self.val_iter = [], [], [], [], [], [], [], [], []
            for iter, batch in enumerate(val_loader):
                _batched_origin_graph, _batched_frag_graph, _batched_motif_graph, _batched_channel_graph, _batched_index_list, _targets, _smiles, _names = batch
                self.val_batched_origin_graph_list.append(_batched_origin_graph)
                self.val_batched_frag_graph_list.append(_batched_frag_graph)
                self.val_batched_motif_graph_list.append(_batched_motif_graph)
                self.val_batched_channel_graph_list.append(_batched_channel_graph)
                self.val_batched_index_list.append(_batched_index_list)
                self.val_targets_list.append(_targets)
                self.val_smiles_list.append(_smiles)
                self.val_names_list.append(_smiles)
                self.val_iter.append(iter)

            self.test_batched_origin_graph_list, self.test_batched_frag_graph_list, self.test_batched_motif_graph_list, \
            self.test_batched_channel_graph_list, self.test_batched_index_list, self.test_targets_list, self.test_smiles_list, \
            self.test_names_list, self.test_iter = [], [], [], [], [], [], [], [], []
            for iter, batch in enumerate(test_loader):
                _batched_origin_graph, _batched_frag_graph, _batched_motif_graph, _batched_channel_graph, _batched_index_list, _targets, _smiles, _names = batch
                self.test_batched_origin_graph_list.append(_batched_origin_graph)
                self.test_batched_frag_graph_list.append(_batched_frag_graph)
                self.test_batched_motif_graph_list.append(_batched_motif_graph)
                self.test_batched_channel_graph_list.append(_batched_channel_graph)
                self.test_batched_index_list.append(_batched_index_list)
                self.test_targets_list.append(_targets)
                self.test_smiles_list.append(_smiles)
                self.test_names_list.append(_smiles)
                self.test_iter.append(iter)

            self.all_batched_origin_graph_list, self.all_batched_frag_graph_list, self.all_batched_motif_graph_list, \
            self.all_batched_channel_graph_list, self.all_batched_index_list, self.all_targets_list, self.all_smiles_list, \
            self.all_names_list, self.all_iter = [], [], [], [], [], [], [], [], []
            for iter, batch in enumerate(raw_loader):
                _batched_origin_graph, _batched_frag_graph, _batched_motif_graph, _batched_channel_graph, _batched_index_list, _targets, _smiles, _names = batch
                self.all_batched_origin_graph_list.append(_batched_origin_graph)
                self.all_batched_frag_graph_list.append(_batched_frag_graph)
                self.all_batched_motif_graph_list.append(_batched_motif_graph)
                self.all_batched_channel_graph_list.append(_batched_channel_graph)
                self.all_batched_index_list.append(_batched_index_list)
                self.all_targets_list.append(_targets)
                self.all_smiles_list.append(_smiles)
                self.all_names_list.append(_smiles)
                self.all_iter.append(iter)

        elif frag == 1:
            self.train_batched_origin_graph_list, self.train_batched_frag_graph_list, self.train_batched_motif_graph_list, self.train_targets_list, self.train_smiles_list, self.train_names_list, self.train_iter = [], [], [], [], [], [], []
            for iter, batch in enumerate(train_loader):
                _batched_origin_graph, _batched_frag_graph, _batched_motif_graph, _targets, _smiles, _names = batch
                self.train_batched_origin_graph_list.append(_batched_origin_graph)
                self.train_batched_frag_graph_list.append(_batched_frag_graph)
                self.train_batched_motif_graph_list.append(_batched_motif_graph)
                self.train_targets_list.append(_targets)
                self.train_smiles_list.append(_smiles)
                self.train_names_list.append(_smiles)
                self.train_iter.append(iter)

            self.val_batched_origin_graph_list, self.val_batched_frag_graph_list, self.val_batched_motif_graph_list, self.val_targets_list, self.val_smiles_list, self.val_names_list, self.val_iter = [], [], [], [], [], [], []
            for iter, batch in enumerate(val_loader):
                _batched_origin_graph, _batched_frag_graph, _batched_motif_graph, _targets, _smiles, _names = batch
                self.val_batched_origin_graph_list.append(_batched_origin_graph)
                self.val_batched_frag_graph_list.append(_batched_frag_graph)
                self.val_batched_motif_graph_list.append(_batched_motif_graph)
                self.val_targets_list.append(_targets)
                self.val_smiles_list.append(_smiles)
                self.val_names_list.append(_smiles)
                self.val_iter.append(iter)

            self.test_batched_origin_graph_list, self.test_batched_frag_graph_list, self.test_batched_motif_graph_list, self.test_targets_list, self.test_smiles_list, self.test_names_list, self.test_iter = [], [], [], [], [], [], []
            for iter, batch in enumerate(test_loader):
                _batched_origin_graph, _batched_frag_graph, _batched_motif_graph, _targets, _smiles, _names = batch
                self.test_batched_origin_graph_list.append(_batched_origin_graph)
                self.test_batched_frag_graph_list.append(_batched_frag_graph)
                self.test_batched_motif_graph_list.append(_batched_motif_graph)
                self.test_targets_list.append(_targets)
                self.test_smiles_list.append(_smiles)
                self.test_names_list.append(_smiles)
                self.test_iter.append(iter)

            self.all_batched_origin_graph_list, self.all_batched_frag_graph_list, self.all_batched_motif_graph_list, self.all_targets_list, self.all_smiles_list, self.all_names_list, self.all_iter = [], [], [], [], [], [], []
            for iter, batch in enumerate(raw_loader):
                _batched_origin_graph, _batched_frag_graph, _batched_motif_graph, _targets, _smiles, _names = batch
                self.all_batched_origin_graph_list.append(_batched_origin_graph)
                self.all_batched_frag_graph_list.append(_batched_frag_graph)
                self.all_batched_motif_graph_list.append(_batched_motif_graph)
                self.all_targets_list.append(_targets)
                self.all_smiles_list.append(_smiles)
                self.all_names_list.append(_smiles)
                self.all_iter.append(iter)

        else:
            self.train_batched_origin_graph_list, self.train_targets_list, self.train_smiles_list, self.train_names_list, self.train_iter = [], [], [], [], []
            for iter, batch in enumerate(train_loader):
                _batched_origin_graph,  _targets, _smiles, _names = batch
                self.train_batched_origin_graph_list.append(_batched_origin_graph)
                self.train_targets_list.append(_targets)
                self.train_smiles_list.append(_smiles)
                self.train_names_list.append(_smiles)
                self.train_iter.append(iter)

            self.val_batched_origin_graph_list, self.val_targets_list, self.val_smiles_list, self.val_names_list, self.val_iter = [], [], [], [], []
            for iter, batch in enumerate(val_loader):
                _batched_origin_graph, _targets, _smiles, _names = batch
                self.val_batched_origin_graph_list.append(_batched_origin_graph)
                self.val_targets_list.append(_targets)
                self.val_smiles_list.append(_smiles)
                self.val_names_list.append(_smiles)
                self.val_iter.append(iter)

            self.test_batched_origin_graph_list, self.test_targets_list, self.test_smiles_list, self.test_names_list, self.test_iter = [], [], [], [], []
            for iter, batch in enumerate(test_loader):
                _batched_origin_graph, _targets, _smiles, _names = batch
                self.test_batched_origin_graph_list.append(_batched_origin_graph)
                self.test_targets_list.append(_targets)
                self.test_smiles_list.append(_smiles)
                self.test_names_list.append(_smiles)
                self.test_iter.append(iter)

            self.all_batched_origin_graph_list, self.all_targets_list, self.all_smiles_list, self.all_names_list, self.all_iter = [], [], [], [], []
            for iter, batch in enumerate(raw_loader):
                _batched_origin_graph, _targets, _smiles, _names = batch
                self.all_batched_origin_graph_list.append(_batched_origin_graph)
                self.all_targets_list.append(_targets)
                self.all_smiles_list.append(_smiles)
                self.all_names_list.append(_smiles)
                self.all_iter.append(iter)


class PreFetch_extra(object):
    def __init__(self, dataset, frag):
        if frag == 2:
            self.batched_origin_graph_list, self.batched_frag_graph_list, self.batched_motif_graph_list, \
            self.batched_channel_graph_list, self.batched_index_list, self.targets_list, self.smiles_list, \
            self.names_list, self.iter = [], [], [], [], [], [], [], [], []
            for iter, batch in enumerate(dataset):
                _batched_origin_graph, _batched_frag_graph, _batched_motif_graph, _batched_channel_graph, _batched_index_list, _targets, _smiles, _names = batch
                self.batched_origin_graph_list.append(_batched_origin_graph)
                self.batched_frag_graph_list.append(_batched_frag_graph)
                self.batched_motif_graph_list.append(_batched_motif_graph)
                self.batched_channel_graph_list.append(_batched_channel_graph)
                self.batched_index_list.append(_batched_index_list)
                self.targets_list.append(_targets)
                self.smiles_list.append(_smiles)
                self.names_list.append(_smiles)
                self.iter.append(iter)

        elif frag == 1:
            self.batched_origin_graph_list, self.batched_frag_graph_list, self.batched_motif_graph_list, self.targets_list, self.smiles_list, self.names_list, self.iter = [], [], [], [], [], [], []
            for iter, batch in enumerate(dataset):
                _batched_origin_graph, _batched_frag_graph, _batched_motif_graph, _targets, _smiles, _names = batch
                self.batched_origin_graph_list.append(_batched_origin_graph)
                self.batched_frag_graph_list.append(_batched_frag_graph)
                self.batched_motif_graph_list.append(_batched_motif_graph)
                self.targets_list.append(_targets)
                self.smiles_list.append(_smiles)
                self.names_list.append(_smiles)
                self.iter.append(iter)

        else:
            self.batched_origin_graph_list, self.targets_list, self.smiles_list, self.names_list, self.iter = [], [], [], [], []
            for iter, batch in enumerate(dataset):
                _batched_origin_graph,  _targets, _smiles, _names = batch
                self.batched_origin_graph_list.append(_batched_origin_graph)
                self.targets_list.append(_targets)
                self.smiles_list.append(_smiles)
                self.names_list.append(_smiles)
                self.iter.append(iter)


def train_epoch(model, optimizer, scaling, iter, batched_origin_graph, targets, smiles, names, n_param=None):
    st = time.time()
    model.train()
    score_list = []
    target_list = []
    epoch_loss = 0
    for i in iter:
        batch_origin_node = batched_origin_graph[i].ndata['feat'].to(device='cuda')
        batch_origin_edge = batched_origin_graph[i].edata['feat'].to(device='cuda')
        batch_origin_global = batched_origin_graph[i].ndata['global_feat'].to(device='cuda')
        batch_origin_graph = batched_origin_graph[i].to(device='cuda')

        optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(False)
        score = model.forward(batch_origin_graph, batch_origin_node, batch_origin_edge)
        target = targets[i].float().to(device='cuda')
        loss = model.loss(score, target)
        loss.backward()
        optimizer.step()
        score_list.append(score)
        target_list.append(target)
        epoch_loss += loss.detach().item()
    score_list = torch.cat(score_list, dim=0)
    target_list = torch.cat(target_list, dim=0)

    epoch_train_metrics = Metrics(scaling.ReScaler(target_list.detach().to(device='cpu')),
                                  scaling.ReScaler(score_list.detach().to(device='cpu')), n_param)
    return model, epoch_loss, epoch_train_metrics


def evaluate(model, scaling, iter, batched_origin_graph, targets, smiles, names, n_param=None, flag=False):
    model.eval()
    score_list = []
    target_list = []
    epoch_loss = 0
    for i in iter:
        batch_origin_node = batched_origin_graph[i].ndata['feat'].to(device='cuda')
        batch_origin_edge = batched_origin_graph[i].edata['feat'].to(device='cuda')
        batch_origin_global = batched_origin_graph[i].ndata['global_feat'].to(device='cuda')
        batch_origin_graph = batched_origin_graph[i].to(device='cuda')

        torch.autograd.set_detect_anomaly(False)
        score = model.forward(batch_origin_graph, batch_origin_node, batch_origin_edge)
        target = targets[i].float().to(device='cuda')
        loss = model.loss(score, target)
        score_list.append(score)
        target_list.append(target)
        epoch_loss += loss.detach().item()
    score_list = torch.cat(score_list, dim=0)
    target_list = torch.cat(target_list, dim=0)

    epoch_eval_metrics = Metrics(scaling.ReScaler(target_list.detach().to(device='cpu')),
                                  scaling.ReScaler(score_list.detach().to(device='cpu')), n_param)

    predict = scaling.ReScaler(score_list.detach().to(device='cpu'))
    true = scaling.ReScaler(target_list.detach().to(device='cpu'))
    if flag:
        return epoch_loss, epoch_eval_metrics, predict, true, smiles
    else:
        return epoch_loss, epoch_eval_metrics


def evaluate_descriptors(model, scaling, iter, batched_origin_graph, targets, smiles, names, n_param=None):
    model.eval()
    for i in iter:
        batch_origin_node = batched_origin_graph[i].ndata['feat'].to(device='cuda')
        batch_origin_edge = batched_origin_graph[i].edata['feat'].to(device='cuda')
        batch_origin_global = batched_origin_graph[i].ndata['global_feat'].to(device='cuda')
        batch_origin_graph = batched_origin_graph[i].to(device='cuda')

        torch.autograd.set_detect_anomaly(False)
        _, descriptors = model.forward(batch_origin_graph, batch_origin_node, batch_origin_edge, get_descriptors=True)
    return smiles, descriptors


def evaluate_attention(model, scaling, iter, batched_origin_graph):
    model.eval()
    score_list = []
    attentions_list = []
    for i in iter:
        batch_origin_node = batched_origin_graph[i].ndata['feat'].to(device='cuda')
        batch_origin_edge = batched_origin_graph[i].edata['feat'].to(device='cuda')
        batch_origin_global = batched_origin_graph[i].ndata['global_feat'].to(device='cuda')
        batch_origin_graph = batched_origin_graph[i].to(device='cuda')

        torch.autograd.set_detect_anomaly(False)
        score, attention = model.forward(batch_origin_graph, batch_origin_node, batch_origin_edge, get_attention=True)
        score_list.append(score)
        attentions_list.extend(attention)
    score_list = torch.cat(score_list, dim=0)

    predictions_list = scaling.ReScaler(score_list.detach().to(device='cpu').numpy())
    return predictions_list, attentions_list

def train_epoch_frag(model, optimizer, scaling, iter, batched_origin_graph, batched_frag_graph, batched_motif_graph, targets, smiles, names, n_param=None):
    st = time.time()
    model.train()
    score_list = []
    target_list = []
    epoch_loss = 0
    for i in iter:
        batch_origin_node = batched_origin_graph[i].ndata['feat'].to(device='cuda')
        batch_origin_edge = batched_origin_graph[i].edata['feat'].to(device='cuda')
        batch_origin_global = batched_origin_graph[i].ndata['global_feat'].to(device='cuda')
        batch_origin_graph = batched_origin_graph[i].to(device='cuda')

        batch_frag_node = batched_frag_graph[i].ndata['feat'].to(device='cuda')
        batch_frag_edge = batched_frag_graph[i].edata['feat'].to(device='cuda')
        batch_frag_graph = batched_frag_graph[i].to(device='cuda')

        batch_motif_node = batched_motif_graph[i].ndata['feat'].to(device='cuda')
        batch_motif_edge = batched_motif_graph[i].edata['feat'].to(device='cuda')
        batch_motif_graph = batched_motif_graph[i].to(device='cuda')

        optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(False)
        if True:
            score = model.forward(batch_origin_graph, batch_origin_node, batch_origin_edge,
                                  batch_frag_graph, batch_frag_node, batch_frag_edge,
                                  batch_motif_graph, batch_motif_node, batch_motif_edge)
        target = targets[i].float().to(device='cuda')
        loss = model.loss(score, target)
        loss.backward()
        optimizer.step()
        score_list.append(score)
        target_list.append(target)
        epoch_loss += loss.detach().item()
    score_list = torch.cat(score_list, dim=0)
    target_list = torch.cat(target_list, dim=0)
    epoch_train_metrics = Metrics(scaling.ReScaler(target_list.detach().to(device='cpu')),
                                  scaling.ReScaler(score_list.detach().to(device='cpu')), n_param)
    return model, epoch_loss, epoch_train_metrics


def train_epoch_gcgat(model, optimizer, scaling, iter, batched_origin_graph, batched_frag_graph, batched_motif_graph, batched_channel_graph, batched_channel_index, targets, smiles, names, n_param=None):
    st = time.time()
    model.train()
    score_list = []
    target_list = []
    epoch_loss = 0
    for i in iter:
        batch_origin_node = batched_origin_graph[i].ndata['feat'].to(device='cuda')
        batch_origin_edge = batched_origin_graph[i].edata['feat'].to(device='cuda')
        batch_origin_global = batched_origin_graph[i].ndata['global_feat'].to(device='cuda')
        batch_origin_graph = batched_origin_graph[i].to(device='cuda')

        batch_frag_node = batched_frag_graph[i].ndata['feat'].to(device='cuda')
        batch_frag_edge = batched_frag_graph[i].edata['feat'].to(device='cuda')
        batch_frag_graph = batched_frag_graph[i].to(device='cuda')

        batch_motif_node = batched_motif_graph[i].ndata['feat'].to(device='cuda')
        batch_motif_edge = batched_motif_graph[i].edata['feat'].to(device='cuda')
        batch_motif_graph = batched_motif_graph[i].to(device='cuda')

        batch_channel_graph = batched_channel_graph[i].to(device='cuda')
        batch_channel_index = batched_channel_index[i]

        optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(False)
        if True:
            score = model.forward(batch_origin_graph, batch_origin_node, batch_origin_edge,
                                  batch_frag_graph, batch_frag_node, batch_frag_edge,
                                  batch_motif_graph, batch_motif_node, batch_motif_edge, batch_channel_graph, batch_channel_index)
        target = targets[i].float().to(device='cuda')
        loss = model.loss(score, target)
        loss.backward()
        optimizer.step()
        score_list.append(score)
        target_list.append(target)
        epoch_loss += loss.detach().item()
    score_list = torch.cat(score_list, dim=0)
    target_list = torch.cat(target_list, dim=0)
    epoch_train_metrics = Metrics(scaling.ReScaler(target_list.detach().to(device='cpu')),
                                  scaling.ReScaler(score_list.detach().to(device='cpu')), n_param)
    return model, epoch_loss, epoch_train_metrics


def evaluate_frag(model, scaling, iter, batched_origin_graph, batched_frag_graph, batched_motif_graph, targets, smiles, names, n_param=None, flag=False):
    model.eval()
    score_list = []
    target_list = []
    epoch_loss = 0
    for i in iter:
        batch_origin_node = batched_origin_graph[i].ndata['feat'].to(device='cuda')
        batch_origin_edge = batched_origin_graph[i].edata['feat'].to(device='cuda')
        batch_origin_global = batched_origin_graph[i].ndata['global_feat'].to(device='cuda')
        batch_origin_graph = batched_origin_graph[i].to(device='cuda')

        batch_frag_node = batched_frag_graph[i].ndata['feat'].to(device='cuda')
        batch_frag_edge = batched_frag_graph[i].edata['feat'].to(device='cuda')
        batch_frag_graph = batched_frag_graph[i].to(device='cuda')

        batch_motif_node = batched_motif_graph[i].ndata['feat'].to(device='cuda')
        batch_motif_edge = batched_motif_graph[i].edata['feat'].to(device='cuda')
        batch_motif_graph = batched_motif_graph[i].to(device='cuda')

        torch.autograd.set_detect_anomaly(False)
        if True:
            score = model.forward(batch_origin_graph, batch_origin_node, batch_origin_edge,
                                  batch_frag_graph, batch_frag_node, batch_frag_edge,
                                  batch_motif_graph, batch_motif_node, batch_motif_edge)
        target = targets[i].float().to(device='cuda')
        loss = model.loss(score, target)
        score_list.append(score)
        target_list.append(target)
        epoch_loss += loss.detach().item()
    score_list = torch.cat(score_list, dim=0)
    target_list = torch.cat(target_list, dim=0)
    epoch_eval_metrics = Metrics(scaling.ReScaler(target_list.detach().to(device='cpu')),
                                  scaling.ReScaler(score_list.detach().to(device='cpu')), n_param)

    predict = scaling.ReScaler(score_list.detach().to(device='cpu'))
    true = scaling.ReScaler(target_list.detach().to(device='cpu'))
    if flag:
        return epoch_loss, epoch_eval_metrics, predict, true, smiles
    else:
        return epoch_loss, epoch_eval_metrics


def evaluate_gcgat(model, scaling, iter, batched_origin_graph, batched_frag_graph, batched_motif_graph, batched_channel_graph, batched_channel_index, targets, smiles, names, n_param=None, flag=False):
    model.eval()
    score_list = []
    target_list = []
    epoch_loss = 0
    for i in iter:
        batch_origin_node = batched_origin_graph[i].ndata['feat'].to(device='cuda')
        batch_origin_edge = batched_origin_graph[i].edata['feat'].to(device='cuda')
        batch_origin_global = batched_origin_graph[i].ndata['global_feat'].to(device='cuda')
        batch_origin_graph = batched_origin_graph[i].to(device='cuda')

        batch_frag_node = batched_frag_graph[i].ndata['feat'].to(device='cuda')
        batch_frag_edge = batched_frag_graph[i].edata['feat'].to(device='cuda')
        batch_frag_graph = batched_frag_graph[i].to(device='cuda')

        batch_motif_node = batched_motif_graph[i].ndata['feat'].to(device='cuda')
        batch_motif_edge = batched_motif_graph[i].edata['feat'].to(device='cuda')
        batch_motif_graph = batched_motif_graph[i].to(device='cuda')

        batch_channel_graph = batched_channel_graph[i].to(device='cuda')
        batch_channel_index = batched_channel_index[i]

        torch.autograd.set_detect_anomaly(False)
        if True:
            score = model.forward(batch_origin_graph, batch_origin_node, batch_origin_edge,
                                  batch_frag_graph, batch_frag_node, batch_frag_edge,
                                  batch_motif_graph, batch_motif_node, batch_motif_edge, batch_channel_graph, batch_channel_index)
        target = targets[i].float().to(device='cuda')
        loss = model.loss(score, target)
        score_list.append(score)
        target_list.append(target)
        epoch_loss += loss.detach().item()
    score_list = torch.cat(score_list, dim=0)
    target_list = torch.cat(target_list, dim=0)
    epoch_eval_metrics = Metrics(scaling.ReScaler(target_list.detach().to(device='cpu')),
                                  scaling.ReScaler(score_list.detach().to(device='cpu')), n_param)

    predict = scaling.ReScaler(score_list.detach().to(device='cpu'))
    true = scaling.ReScaler(target_list.detach().to(device='cpu'))
    if flag:
        return epoch_loss, epoch_eval_metrics, predict, true, smiles
    else:
        return epoch_loss, epoch_eval_metrics


def evaluate_frag_descriptors(model, scaling, iter, batched_origin_graph, batched_frag_graph, batched_motif_graph, targets, smiles, names, n_param=None):
    model.eval()
    score_list = []
    target_list = []
    for i in iter:
        batch_origin_node = batched_origin_graph[i].ndata['feat'].to(device='cuda')
        batch_origin_edge = batched_origin_graph[i].edata['feat'].to(device='cuda')
        batch_origin_global = batched_origin_graph[i].ndata['global_feat'].to(device='cuda')
        batch_origin_graph = batched_origin_graph[i].to(device='cuda')

        batch_frag_node = batched_frag_graph[i].ndata['feat'].to(device='cuda')
        batch_frag_edge = batched_frag_graph[i].edata['feat'].to(device='cuda')
        batch_frag_graph = batched_frag_graph[i].to(device='cuda')

        batch_motif_node = batched_motif_graph[i].ndata['feat'].to(device='cuda')
        batch_motif_edge = batched_motif_graph[i].edata['feat'].to(device='cuda')
        batch_motif_graph = batched_motif_graph[i].to(device='cuda')

        torch.autograd.set_detect_anomaly(False)
        if True:
            _, descriptors = model.forward(batch_origin_graph, batch_origin_node, batch_origin_edge,
                                  batch_frag_graph, batch_frag_node, batch_frag_edge,
                                  batch_motif_graph, batch_motif_node, batch_motif_edge, get_descriptors=True)

    return smiles, descriptors


def evaluate_gcgat_descriptors(model, scaling, iter, batched_origin_graph, batched_frag_graph, batched_motif_graph, batched_channel_graph, batched_channel_index, targets, smiles, names, n_param=None):
    model.eval()
    score_list = []
    target_list = []
    for i in iter:
        batch_origin_node = batched_origin_graph[i].ndata['feat'].to(device='cuda')
        batch_origin_edge = batched_origin_graph[i].edata['feat'].to(device='cuda')
        batch_origin_global = batched_origin_graph[i].ndata['global_feat'].to(device='cuda')
        batch_origin_graph = batched_origin_graph[i].to(device='cuda')

        batch_frag_node = batched_frag_graph[i].ndata['feat'].to(device='cuda')
        batch_frag_edge = batched_frag_graph[i].edata['feat'].to(device='cuda')
        batch_frag_graph = batched_frag_graph[i].to(device='cuda')

        batch_motif_node = batched_motif_graph[i].ndata['feat'].to(device='cuda')
        batch_motif_edge = batched_motif_graph[i].edata['feat'].to(device='cuda')
        batch_motif_graph = batched_motif_graph[i].to(device='cuda')

        batch_channel_graph = batched_channel_graph[i].to(device='cuda')
        batch_channel_index = batched_channel_index[i]

        torch.autograd.set_detect_anomaly(False)
        if True:
            _, descriptors = model.forward(batch_origin_graph, batch_origin_node, batch_origin_edge,
                                  batch_frag_graph, batch_frag_node, batch_frag_edge,
                                  batch_motif_graph, batch_motif_node, batch_motif_edge, batch_channel_graph, batch_channel_index, get_descriptors=True)

    return smiles, descriptors


def evaluate_frag_attention(model, scaling, iter, batched_origin_graph, batched_frag_graph, batched_motif_graph):
    model.eval()
    score_list = []
    attentions_list = []
    for i in iter:
        batch_origin_node = batched_origin_graph[i].ndata['feat'].to(device='cuda')
        batch_origin_edge = batched_origin_graph[i].edata['feat'].to(device='cuda')
        batch_origin_global = batched_origin_graph[i].ndata['global_feat'].to(device='cuda')
        batch_origin_graph = batched_origin_graph[i].to(device='cuda')

        batch_frag_node = batched_frag_graph[i].ndata['feat'].to(device='cuda')
        batch_frag_edge = batched_frag_graph[i].edata['feat'].to(device='cuda')
        batch_frag_graph = batched_frag_graph[i].to(device='cuda')

        batch_motif_node = batched_motif_graph[i].ndata['feat'].to(device='cuda')
        batch_motif_edge = batched_motif_graph[i].edata['feat'].to(device='cuda')
        batch_motif_graph = batched_motif_graph[i].to(device='cuda')

        torch.autograd.set_detect_anomaly(False)
        if True:
            score, attention = model.forward(batch_origin_graph, batch_origin_node, batch_origin_edge,
                                  batch_frag_graph, batch_frag_node, batch_frag_edge,
                                  batch_motif_graph, batch_motif_node, batch_motif_edge, get_descriptors=True, get_attention=True)
        score_list.append(score)
        attentions_list.extend(attention)
    score_list = torch.cat(score_list, dim=0)

    predictions_list = scaling.ReScaler(score_list.detach().to(device='cpu').numpy())
    return predictions_list, attentions_list


def evaluate_gcgat_attention(model, scaling, iter, batched_origin_graph, batched_frag_graph, batched_motif_graph, batched_channel_graph, batched_channel_index):
    model.eval()
    score_list = []
    attentions_list = []
    for i in iter:
        batch_origin_node = batched_origin_graph[i].ndata['feat'].to(device='cuda')
        batch_origin_edge = batched_origin_graph[i].edata['feat'].to(device='cuda')
        batch_origin_global = batched_origin_graph[i].ndata['global_feat'].to(device='cuda')
        batch_origin_graph = batched_origin_graph[i].to(device='cuda')

        batch_frag_node = batched_frag_graph[i].ndata['feat'].to(device='cuda')
        batch_frag_edge = batched_frag_graph[i].edata['feat'].to(device='cuda')
        batch_frag_graph = batched_frag_graph[i].to(device='cuda')

        batch_motif_node = batched_motif_graph[i].ndata['feat'].to(device='cuda')
        batch_motif_edge = batched_motif_graph[i].edata['feat'].to(device='cuda')
        batch_motif_graph = batched_motif_graph[i].to(device='cuda')

        batch_channel_graph = batched_channel_graph[i].to(device='cuda')
        batch_channel_index = batched_channel_index[i]

        torch.autograd.set_detect_anomaly(False)
        if True:
            score, attention = model.forward(batch_origin_graph, batch_origin_node, batch_origin_edge,
                                  batch_frag_graph, batch_frag_node, batch_frag_edge,
                                  batch_motif_graph, batch_motif_node, batch_motif_edge, batch_channel_graph, batch_channel_index, get_descriptors=True, get_attention=True)
        score_list.append(score)
        attentions_list.extend(attention)
    score_list = torch.cat(score_list, dim=0)

    predictions_list = scaling.ReScaler(score_list.detach().to(device='cpu').numpy())
    return predictions_list, attentions_list

