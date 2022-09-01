"""
======================================================================================================================
0. Introduction
This script is the architecture of Directed Message-Passing Neural Network (D-MPNN).
The idea is introduced in
'Are Learned Molecular Representations Ready For Prime Time?'
https://arxiv.org/pdf/1904.01561v2.pdf

By FAN FAN (s192217@dtu.dk)
======================================================================================================================
1. Load Package
Loading the necessary package:
dgl, torch
======================================================================================================================
2. Steps
Class 'DMPNNet' is the main network function where data is fed in and generate output.

Class 'DMPNNLayer' is the basic layer building this neural network.
======================================================================================================================
"""
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import Set2Set
import dgl.function as fn

from .Set2Set import Set2Set
import seaborn as sns
import matplotlib.pyplot as plt

class DMPNNLayer(nn.Module):
    def __init__(self, net_params):
        super(DMPNNLayer, self).__init__()
        self.hidden_dim = net_params['hidden_dim']
        self.W_m = nn.Linear(net_params['hidden_dim'], net_params['hidden_dim'], bias=True)
        self.dropout_layer = nn.Dropout(net_params['dropout'])
        self.reset_parameters()

    def reset_parameters(self):
        self.W_m.reset_parameters()

    def message_func(self, edges):
        return {'edge_feats': edges.data['edge_feats']}

    def reducer_sum(self, nodes):
        return {'full_feats': torch.sum(nodes.mailbox['edge_feats'], 1)}

    def edge_func(self, edges):
        return {'new_edge_feats': edges.dst['full_feats'] - edges.data['edge_feats']}

    def forward(self, graph, node, inputs, hidden_feats):
        with graph.local_scope():
            graph.ndata['node_feats'] = node
            graph.edata['edge_feats'] = hidden_feats
            graph.update_all(self.message_func, self.reducer_sum)
            graph.apply_edges(self.edge_func)
            hidden_feats = F.relu(inputs + self.W_m(graph.edata['new_edge_feats']))
            hidden_feats = self.dropout_layer(hidden_feats)
            return hidden_feats


class DMPNNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.depth = net_params['depth']
        self.embedding_node_lin = nn.Sequential(
            nn.Linear(net_params['num_atom_type'], net_params['hidden_dim'], bias=True),
            nn.LeakyReLU()
        )
        self.embedding_edge_lin = nn.Sequential(
            nn.Linear(net_params['num_bond_type'], net_params['hidden_dim'], bias=True),
            nn.LeakyReLU()
        )
        self.init_h_func = nn.Sequential(
            nn.Linear(2 * net_params['hidden_dim'], net_params['hidden_dim'], bias=True),
            nn.ReLU()
        )
        self.W_a = nn.Sequential(
            nn.Linear(2 * net_params['hidden_dim'], net_params['hidden_dim'], bias=True),
            nn.ReLU()
        )
        self.dropout_layer = nn.Dropout(net_params['dropout'])
        self.DMPNNLayer = nn.ModuleList([DMPNNLayer(net_params) for _ in range(self.depth)])
        self.linear_prediction = nn.Sequential(
                nn.Linear(net_params['hidden_dim'], int(net_params['hidden_dim']/2), bias=True),
                nn.ReLU(),
                nn.Linear(int(net_params['hidden_dim']/2), 1, bias=True))
        self.reset_parameters()

    def reset_parameters(self):
        self.init_h_func[0].reset_parameters()
        self.W_a[0].reset_parameters()
        self.embedding_node_lin[0].reset_parameters()
        self.embedding_edge_lin[0].reset_parameters()
        for layer in self.linear_prediction:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def send_income_edge(self, edges):
        return {'mail': edges.data['feat']}

    def sum_income_edge(self, nodes):
        hidden_feats = self.W_a(torch.cat([nodes.data['feat'], torch.sum(nodes.mailbox['mail'], 1)], dim=-1))
        hidden_feats = self.dropout_layer(hidden_feats)
        return {'hidden_feats': hidden_feats}

    def forward(self, graph, node, edge, get_descriptors=False):
        node = node.float()
        edge = edge.float()
        node = self.embedding_node_lin(node)
        edge = self.embedding_edge_lin(edge)
        src_node_id = graph.edges()[0]
        #dst_node_id = graph.edges()[1]
        hidden_feats = self.init_h_func(torch.cat([node[src_node_id], edge], dim=-1)) # hidden states include the source node and edge features, num_bonds x num_features
        inputs = hidden_feats
        for layer in self.DMPNNLayer:
            hidden_feats = layer(graph, node, inputs, hidden_feats)

        graph.edata['feat'] = hidden_feats
        graph.ndata['feat'] = node
        graph.update_all(self.send_income_edge, self.sum_income_edge)
        graph_feats = dgl.sum_nodes(graph, 'hidden_feats')
        output = self.linear_prediction(graph_feats)
        if get_descriptors:
            return output, graph_feats
        else:
            return output

    def loss(self, scores, targets):
        loss = nn.MSELoss()(scores, targets)
        return loss


