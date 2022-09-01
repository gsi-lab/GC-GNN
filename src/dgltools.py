# -*- coding: utf-8 -*-
# @Time    : 2022/3/23 13:17
# @Author  : FAN FAN
# @Site    : 
# @File    : dgltools.py
# @Software: PyCharm
import numpy as np
import torch
import dgl


def collate_molgraphs(samples):
    origin_graphs, targets, smiles, names = map(list, zip(*samples))
    targets = torch.tensor(np.array(targets)).unsqueeze(1)
    batched_origin_graph = dgl.batch(origin_graphs)
    return batched_origin_graph, targets, smiles, names


def collate_fraggraphs(samples):
    # origin_graphs, motif_graphs: list of graphs:
    origin_graphs, frag_graphs, motif_graphs, _, targets, smiles, names = map(list, zip(*samples))
    targets = torch.tensor(np.array(targets)).unsqueeze(1)

    batched_origin_graph = dgl.batch(origin_graphs)
    batched_frag_graph = dgl.batch(frag_graphs)
    batched_motif_graph = dgl.batch(motif_graphs)
    return batched_origin_graph, batched_frag_graph, batched_motif_graph, targets, smiles, names


def collate_gcgatgraphs(samples):
    # origin_graphs, motif_graphs: list of graphs:
    origin_graphs, frag_graphs, motif_graphs, channel_graphs, targets, smiles, names = map(list, zip(*samples))
    targets = torch.tensor(np.array(targets)).unsqueeze(1)

    batched_origin_graph = dgl.batch(origin_graphs)
    batched_frag_graph = dgl.batch(frag_graphs)
    batched_motif_graph = dgl.batch(motif_graphs)
    batched_channel_graph = dgl.batch(channel_graphs)

    batched_index_list = []
    batch_len = batched_channel_graph.batch_size
    for i in range(batch_len):
        batched_index_list.append(i)
        batched_index_list.append(i + batch_len)
        batched_index_list.append(i + 2 * batch_len)

    return batched_origin_graph, batched_frag_graph, batched_motif_graph, batched_channel_graph, batched_index_list, targets, smiles, names


def collate_fraggraphs_backup(samples):
    # origin_graphs, motif_graphs: list of graphs:
    origin_graphs, frag_graphs, motif_graphs, targets, smiles, names = map(list, zip(*samples))
    targets = torch.tensor(np.array(targets)).unsqueeze(1)
    frag_graphs_list = []
    for item in frag_graphs:
        frag_graphs_list.extend(item)

    batched_origin_graph = dgl.batch(origin_graphs)
    batched_frag_graph = dgl.batch(frag_graphs_list)
    batched_motif_graph = dgl.batch(motif_graphs)
    return batched_origin_graph, batched_frag_graph, batched_motif_graph, targets, smiles, names

