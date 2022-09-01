"""
======================================================================================================================
0. Introduction
This script is the architecture of Message-Passing Neural Network (MPNN).
The idea is introduced in
'Neural Message Passing for Quantum Chemistry.'
https://arxiv.org/abs/1704.01212v2

By FAN FAN (s192217@dtu.dk)
======================================================================================================================
1. Load Package
Loading the necessary package:
dgl, torch
======================================================================================================================
2. Steps
Class 'MPNNet' is the main network function where data is fed in and generate output.

Class 'MPNNLayer' is the basic layer building this neural network.
======================================================================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import Set2Set

from .Set2Set import Set2Set


class MPNNLayer(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.hidden_dim = net_params['hidden_dim']
        self.is_batch_norm = net_params['batch_norm']
        self.is_layer_norm = net_params['layer_norm']
        if net_params['batch_norm']:
            self.batch_norm = nn.BatchNorm1d(net_params['hidden_dim'])
        if net_params['layer_norm']:
            self.layer_norm = nn.LayerNorm(net_params['hidden_dim'])
        self.edge_network = nn.Sequential(
            nn.Linear(net_params['hidden_dim'], 2 * net_params['hidden_dim'], bias=True),
            nn.ReLU(),
            nn.Linear(2 * net_params['hidden_dim'], net_params['hidden_dim'] * net_params['hidden_dim'], bias=True)
        )
        self.GRUCell = nn.GRUCell(net_params['hidden_dim'], net_params['hidden_dim'])
        self.reset_parameters()

    def reset_parameters(self):
        #gain = init.calculate_gain('relu')
        self.GRUCell.reset_parameters()
        for layer in self.edge_network:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def message_func(self, edges):
        return {'mail': torch.matmul(edges.src['node_feats'].unsqueeze(1), edges.data['weight']).squeeze(1)}

    def reducer_sum(self, nodes):
        return {'new_node_feats': torch.sum(nodes.mailbox['mail'], 1)}

    def forward(self, graph, node, edge, hidden_feats):
        with graph.local_scope():
            node = node.unsqueeze(-1) if node.dim() == 1 else node
            edge = edge.unsqueeze(-1) if edge.dim() == 1 else edge
            graph.ndata['node_feats'] = node
            graph.edata['weight'] = self.edge_network(edge).view(-1, self.hidden_dim, self.hidden_dim)
            graph.update_all(self.message_func, self.reducer_sum)
            if self.is_batch_norm:
                new_node_feats = F.relu(self.batch_norm(graph.ndata.pop('new_node_feats')))
            else:
                new_node_feats = F.relu(graph.ndata.pop('new_node_feats'))
            if self.is_layer_norm:
                output = self.layer_norm(self.GRUCell(new_node_feats, hidden_feats))
            else:
                output = self.GRUCell(new_node_feats, hidden_feats)
        return output.squeeze(0)


class MPNNNet(nn.Module):
    def __init__(self, net_params):
        super(MPNNNet, self).__init__()
        self.depth = net_params['depth']
        self.residual = net_params['residual']
        self.embedding_node_lin = nn.Sequential(
            nn.Linear(net_params['num_atom_type'], net_params['hidden_dim'], bias=True),
            nn.LeakyReLU()
        )
        self.embedding_edge_lin = nn.Sequential(
            nn.Linear(net_params['num_bond_type'], net_params['hidden_dim'], bias=True),
            nn.LeakyReLU()
        )
        self.MPNNLayer = nn.ModuleList([MPNNLayer(net_params) for _ in range(self.depth)])
        if self.residual:
            self.set2set = Set2Set((self.depth+1) * net_params['hidden_dim'], net_params['device'])
            self.linear_prediction = nn.Sequential(
                nn.Linear(2 * (self.depth+1) * net_params['hidden_dim'], net_params['hidden_dim'], bias=True),
                nn.ReLU(),
                nn.Linear(net_params['hidden_dim'], 1, bias=True)
            )
            self.linear_gate = nn.Linear(2 * net_params['hidden_dim'], net_params['hidden_dim'], bias=True)
        else:
            self.set2set = Set2Set(net_params['hidden_dim'], net_params['device'])
            self.linear_prediction = nn.Sequential(
                nn.Linear(2 * net_params['hidden_dim'], net_params['hidden_dim'], bias=True),
                nn.ReLU(),
                nn.Linear(net_params['hidden_dim'], 1, bias=True)
            )
        self.reset_parameters()

    def reset_parameters(self):
        self.embedding_node_lin[0].reset_parameters()
        self.embedding_edge_lin[0].reset_parameters()
        for layer in self.linear_prediction:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, graph, node, edge, get_descriptors=False):
        """Graph-level regression
        Parameters
        ----------
        graph : DGLGraph
            DGLGraph for a batch of graphs.
        node : float tensor of shape (V, F_n)
            Input node features. V for the number of nodes, F_n for the size of node features.
        edge : float tensor of shape (E, F_e)
            Input edge features. E for the number of edges, F_e for the size of edge features.

        Returns
        ----------
        output : float tensor of shape (G, 1)
            Prediction for the graphs in the batch. G for the number of graphs.
        """
        node = node.float()
        edge = edge.float()
        node = self.embedding_node_lin(node)
        edge = self.embedding_edge_lin(edge)
        hidden_feats = node
        hidden_in = hidden_feats
        hidden_out = node
        for layer in self.MPNNLayer:
            node = layer(graph, node, edge, hidden_feats)
            if self.residual:
                z = torch.sigmoid(self.linear_gate(torch.cat([node, hidden_in], dim=-1)))
                node = z * node + (torch.ones_like(z) - z) * hidden_in
                hidden_out = torch.cat([node, hidden_out], dim=-1)
        if self.residual:
            graph_feats = self.set2set(graph, hidden_out)
        else:
            graph_feats = self.set2set(graph, node)
        output = self.linear_prediction(graph_feats)
        if get_descriptors:
            return output, graph_feats
        else:
            return output

    def loss(self, scores, targets):
        loss = nn.MSELoss()(scores, targets)
        return loss

