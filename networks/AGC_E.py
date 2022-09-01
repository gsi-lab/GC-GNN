"""
======================================================================================================================
0. Introduction
This script is the architecture of Attentive Group Contribution (AGC), a novel GNN combines AttentiveFP with prior knowledge in fragmentation.
Also the implementation of three-scales improve the performance of prediction.

By FAN FAN (s192217@dtu.dk)
======================================================================================================================
1. Load Package
Loading the necessary package:
dgl, torch
======================================================================================================================
2. Steps
Class 'NewFraGATNet' is the network function where data is fed in and generate output,
integrating three blocks 'SingleHeadOriginLayer', 'SingleHeadFragmentLayer' and 'SingleHeadJunctionLayer'.

These three blocks provide the predictions on three-scales,
atom-level (original), fragment-level (fragment), molecule-level (motif/Junction).
======================================================================================================================
"""
import dgl
import torch
import torch.nn as nn

from .AttentiveFP import Atom_AttentiveFP, Mol_AttentiveFP


class SingleHeadFragmentLayer(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.AtomEmbedding = Atom_AttentiveFP(net_params)
        self.FragEmbedding = Mol_AttentiveFP(net_params)
        self.reset_parameters()

    def reset_parameters(self):
        self.AtomEmbedding.reset_parameters()
        self.FragEmbedding.reset_parameters()

    def forward(self, frag_graph, frag_node, frag_edge):
        # node_fragments: tensor: size(num_nodes_in_batch, num_features)
        node_fragments = self.AtomEmbedding(frag_graph, frag_node, frag_edge)
        super_frag, _ = self.FragEmbedding(frag_graph, node_fragments)
        return super_frag


class SingleHeadJunctionLayer(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.project_motif = nn.Sequential(
            nn.Linear(net_params['hidden_dim'] + net_params['hidden_dim'], net_params['hidden_dim'], bias=True)
        )
        self.MotifEmbedding = Atom_AttentiveFP(net_params)
        self.GraphEmbedding = Mol_AttentiveFP(net_params)
        self.reset_parameters()

    def reset_parameters(self):
        for l in self.project_motif:
            if isinstance(l, nn.Linear):
                l.reset_parameters()
        self.MotifEmbedding.reset_parameters()
        self.GraphEmbedding.reset_parameters()

    def forward(self, motif_graph, motif_node, motif_edge):
        motif_node = self.project_motif(motif_node)
        new_motif_node = self.MotifEmbedding(motif_graph, motif_node, motif_edge)
        super_new_graph, super_attention_weight = self.GraphEmbedding(motif_graph, new_motif_node)
        return super_new_graph, super_attention_weight


class AGCNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.embedding_frag_node_lin = nn.Sequential(
            nn.Linear(net_params['num_atom_type'], net_params['hidden_dim'], bias=True),
            nn.LeakyReLU()
        )
        self.embedding_frag_edge_lin = nn.Sequential(
            nn.Linear(net_params['num_bond_type'], net_params['hidden_dim'], bias=True),
            nn.LeakyReLU()
        )
        self.embedding_motif_node_lin = nn.Sequential(
            nn.Linear(net_params['frag_dim'], net_params['hidden_dim'], bias=True),
            nn.LeakyReLU()
        )
        self.embedding_motif_edge_lin = nn.Sequential(
            nn.Linear(net_params['num_bond_type'], net_params['hidden_dim'], bias=True),
            nn.LeakyReLU()
        )

        self.num_heads = net_params['num_heads']
        self.fragment_heads = nn.ModuleList([SingleHeadFragmentLayer(net_params) for _ in range(self.num_heads)])
        self.junction_heads = nn.ModuleList([SingleHeadJunctionLayer(net_params) for _ in range(self.num_heads)])

        self.frag_attend = nn.Sequential(
            nn.Linear(self.num_heads * net_params['hidden_dim'], net_params['hidden_dim'], bias=True),
            nn.ReLU()
        )
        self.motif_attend = nn.Sequential(
            nn.Linear(self.num_heads * net_params['hidden_dim'], net_params['hidden_dim'], bias=True),
            nn.ReLU()
        )

        self.linear_predict = nn.Sequential(
            nn.Dropout(net_params['dropout']),
            nn.Linear(net_params['hidden_dim'], int(net_params['hidden_dim'] / 2), bias=True),
            nn.LeakyReLU(),
            nn.Linear(int(net_params['hidden_dim'] / 2), 1, bias=True)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for fragment_layer in self.fragment_heads:
            fragment_layer.reset_parameters()
        for junction_layer in self.junction_heads:
            junction_layer.reset_parameters()
        for layer in self.linear_predict:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, origin_graph, origin_node, origin_edge, frag_graph, frag_node, frag_edge, motif_graph, motif_node, motif_edge):
        """Graph-level regression
        Parameters
        ----------
        origin_graph : DGLGraph
            DGLGraph for a batch of graphs (original).
        origin_node : float tensor of shape (V, F_n)
            Input node features in origin_graph. V for the number of nodes, F_n for the size of node features.
        origin_edge : float tensor of shape (E, F_e)
            Input edge features in origin_graph. E for the number of edges, F_e for the size of edge features.
        frag_graph : DGLGraph
            DGLGraph for a batch of graphs (fragment).
        frag_node : float tensor of shape (V, F_n)
            Input node features in fragments. V for the number of nodes, F_n for the size of node features.
        frag_edge : float tensor of shape (E', F_e)
            Input edge features in fragments. E' for the number of edges in fragments, F_e for the size of edge features.
        motif_graph : DGLGraph
            DGLGraph for a batch of graphs (motif).
        motif_node : float tensor of shape (V_f, F_n)
            Input node features in motif_graph. V_f for the number of fragments, F_n for the size of node features.
        motif_edge : float tensor of shape (E_f, F_e)
            Input edge features in motif_graph. E_f for the number of edges connecting fragments, F_e for the size of edge features.

        Returns
        ----------
        output : float tensor of shape (G, 1)
            Prediction for the graphs in the batch. G for the number of graphs.
        motif_graph : DGLGraph
            DGLGraph for a batch of graphs with attribute 'attention_weights' of fragments.
        """
        # Fragments Layer:
        frag_node = frag_node.float()
        frag_edge = frag_edge.float()
        frag_node = self.embedding_frag_node_lin(frag_node)
        frag_edge = self.embedding_frag_edge_lin(frag_edge)
        frag_heads_out = [frag_block(frag_graph, frag_node, frag_edge) for frag_block in self.fragment_heads]
        graph_motif = self.frag_attend(torch.cat(frag_heads_out, dim=-1))
        motif_graph.ndata['feats'] = graph_motif
        # Junction Tree Layer:
        #motif_node = motif_node.float()
        motif_edge = motif_edge.float()
        motif_node = self.embedding_motif_node_lin(motif_node)
        motif_edge = self.embedding_motif_edge_lin(motif_edge)
        motif_node = torch.cat([graph_motif, motif_node], dim=-1)
        junction_graph_heads_out = []
        junction_attention_heads_out = []
        for single_head in self.junction_heads:
            single_head_new_graph, single_head_attention_weight = single_head(motif_graph, motif_node, motif_edge)
            junction_graph_heads_out.append(single_head_new_graph)
            junction_attention_heads_out.append(single_head_attention_weight)

        super_new_graph = torch.relu(torch.mean(torch.stack(junction_graph_heads_out, dim=1), dim=1))
        super_attention_weight = torch.mean(torch.stack(junction_attention_heads_out, dim=1), dim=1)
        output = self.linear_predict(super_new_graph)
        motif_graph.ndata['attention_weight'] = super_attention_weight
        #return output, motif_graph
        return output

    def loss(self, scores, targets):
        loss = nn.MSELoss()(scores, targets)
        return loss



