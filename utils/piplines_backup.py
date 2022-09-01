# -*- coding: utf-8 -*-
# @Time    : 2022/3/23 16:41
# @Author  : FAN FAN
# @Site    : 
# @File    : piplines.py
# @Software: PyCharm
import time
import torch
from .metrics import Metrics
from .Set_Seed_Reproducibility import set_seed


def train_epoch(model, optimizer, scaling, dataloader, n_param=None):
    st = time.time()
    model.train()
    score_list = []
    target_list = []
    epoch_loss = 0
    for iter, batch in enumerate(dataloader):
        batched_origin_graph, targets, smiles, names = batch
        batch_origin_node = batched_origin_graph.ndata['feat'].to(device='cuda')
        batch_origin_edge = batched_origin_graph.edata['feat'].to(device='cuda')
        batch_origin_global = batched_origin_graph.ndata['global_feat'].to(device='cuda')
        batch_origin_graph = batched_origin_graph.to(device='cuda')

        optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(False)
        score = model.forward(batch_origin_graph, batch_origin_node, batch_origin_edge)
        target = targets.float().to(device='cuda')
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


def evaluate(model, scaling, dataloader, n_param=None):
    model.eval()
    score_list = []
    target_list = []
    epoch_loss = 0
    for iter, batch in enumerate(dataloader):
        batched_origin_graph, targets, smiles, names = batch
        batch_origin_node = batched_origin_graph.ndata['feat'].to(device='cuda')
        batch_origin_edge = batched_origin_graph.edata['feat'].to(device='cuda')
        batch_origin_global = batched_origin_graph.ndata['global_feat'].to(device='cuda')
        batch_origin_graph = batched_origin_graph.to(device='cuda')

        torch.autograd.set_detect_anomaly(False)
        score = model.forward(batch_origin_graph, batch_origin_node, batch_origin_edge)
        target = targets.float().to(device='cuda')
        loss = model.loss(score, target)
        score_list.append(score)
        target_list.append(target)
        epoch_loss += loss.detach().item()
    score_list = torch.cat(score_list, dim=0)
    target_list = torch.cat(target_list, dim=0)

    epoch_eval_metrics = Metrics(scaling.ReScaler(target_list.detach().to(device='cpu')),
                                  scaling.ReScaler(score_list.detach().to(device='cpu')), n_param)
    return epoch_loss, epoch_eval_metrics


def train_epoch_frag(model, optimizer, scaling, dataloader, n_param=None):
    st = time.time()
    model.train()
    score_list = []
    target_list = []
    epoch_loss = 0
    for iter, batch in enumerate(dataloader):
        batched_origin_graph, batched_frag_graph, batched_motif_graph, targets, smiles, names = batch
        batch_origin_node = batched_origin_graph.ndata['feat'].to(device='cuda')
        batch_origin_edge = batched_origin_graph.edata['feat'].to(device='cuda')
        batch_origin_global = batched_origin_graph.ndata['global_feat'].to(device='cuda')
        batch_origin_graph = batched_origin_graph.to(device='cuda')

        batch_frag_node = batched_frag_graph.ndata['feat'].to(device='cuda')
        batch_frag_edge = batched_frag_graph.edata['feat'].to(device='cuda')
        batch_frag_graph = batched_frag_graph.to(device='cuda')

        batch_motif_node = batched_motif_graph.ndata['feat'].to(device='cuda')
        batch_motif_edge = batched_motif_graph.edata['feat'].to(device='cuda')
        batch_motif_graph = batched_motif_graph.to(device='cuda')

        optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(False)
        score = model.forward(batch_origin_graph, batch_origin_node, batch_origin_edge,
                              batch_frag_graph, batch_frag_node, batch_frag_edge,
                              batch_motif_graph, batch_motif_node, batch_motif_edge)
        target = targets.float().to(device='cuda')
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


def evaluate_frag(model, scaling, dataloader, n_param=None):
    model.eval()
    score_list = []
    target_list = []
    epoch_loss = 0
    for iter, batch in enumerate(dataloader):
        batched_origin_graph, batched_frag_graph, batched_motif_graph, targets, smiles, names = batch
        batch_origin_node = batched_origin_graph.ndata['feat'].to(device='cuda')
        batch_origin_edge = batched_origin_graph.edata['feat'].to(device='cuda')
        batch_origin_global = batched_origin_graph.ndata['global_feat'].to(device='cuda')
        batch_origin_graph = batched_origin_graph.to(device='cuda')

        batch_frag_node = batched_frag_graph.ndata['feat'].to(device='cuda')
        batch_frag_edge = batched_frag_graph.edata['feat'].to(device='cuda')
        batch_frag_graph = batched_frag_graph.to(device='cuda')

        batch_motif_node = batched_motif_graph.ndata['feat'].to(device='cuda')
        batch_motif_edge = batched_motif_graph.edata['feat'].to(device='cuda')
        batch_motif_graph = batched_motif_graph.to(device='cuda')

        torch.autograd.set_detect_anomaly(False)
        score = model.forward(batch_origin_graph, batch_origin_node, batch_origin_edge,
                              batch_frag_graph, batch_frag_node, batch_frag_edge,
                              batch_motif_graph, batch_motif_node, batch_motif_edge)
        target = targets.float().to(device='cuda')
        loss = model.loss(score, target)
        score_list.append(score)
        target_list.append(target)
        epoch_loss += loss.detach().item()
    score_list = torch.cat(score_list, dim=0)
    target_list = torch.cat(target_list, dim=0)

    epoch_eval_metrics = Metrics(scaling.ReScaler(target_list.detach().to(device='cpu')),
                                  scaling.ReScaler(score_list.detach().to(device='cpu')), n_param)
    return epoch_loss, epoch_eval_metrics