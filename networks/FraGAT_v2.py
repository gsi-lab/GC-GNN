"""
======================================================================================================================
0. Introduction
This script is the architecture of GFraGAT, a novel GNN combines AttentiveFP with prior knowledge in fragmentation.
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
import torch.nn.functional as F

from .AttentiveFP import Atom_AttentiveFP, Mol_AttentiveFP


class SingleHeadOriginLayer(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.AtomEmbedding = Atom_AttentiveFP(net_params)
        self.MolEmbedding = Mol_AttentiveFP(net_params)

    def forward(self, origin_graph, origin_node, origin_edge):
        node_origin = self.AtomEmbedding(origin_graph, origin_node, origin_edge)
        super_node_origin, _ = self.MolEmbedding(origin_graph, node_origin)
        return super_node_origin


class SingleHeadFragmentLayer(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.AtomEmbedding = Atom_AttentiveFP(net_params)
        self.FragEmbedding = Mol_AttentiveFP(net_params)

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

    def forward(self, motif_graph, motif_node, motif_edge):
        motif_node = self.project_motif(motif_node)
        new_motif_node = self.MotifEmbedding(motif_graph, motif_node, motif_edge)
        super_new_graph, super_attention_weight = self.GraphEmbedding(motif_graph, new_motif_node)
        return super_new_graph, super_attention_weight


class NewFraGATNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.embedding_node_lin = nn.Sequential(
            nn.Linear(net_params['num_atom_type'], net_params['hidden_dim'], bias=True),
            nn.BatchNorm1d(net_params['hidden_dim']),
            nn.LeakyReLU()
        )
        self.embedding_edge_lin = nn.Sequential(
            nn.Linear(net_params['num_bond_type'], net_params['hidden_dim'], bias=True),
            nn.BatchNorm1d(net_params['hidden_dim']),
            nn.LeakyReLU()
        )
        self.embedding_frag_lin = nn.Sequential(
            nn.Linear(net_params['frag_dim'], net_params['hidden_dim'], bias=True),
            nn.BatchNorm1d(net_params['hidden_dim']),
            nn.LeakyReLU()
        )

        self.num_heads = net_params['num_heads']
        self.origin_heads = nn.ModuleList([SingleHeadOriginLayer(net_params) for _ in range(self.num_heads)])
        self.fragment_heads = nn.ModuleList([SingleHeadFragmentLayer(net_params) for _ in range(self.num_heads)])
        self.junction_heads = nn.ModuleList([SingleHeadJunctionLayer(net_params) for _ in range(self.num_heads)])


        self.origin_attend = nn.Sequential(
            nn.Linear(self.num_heads * net_params['hidden_dim'], net_params['hidden_dim'], bias=True),
            nn.BatchNorm1d(net_params['hidden_dim']),
            nn.ReLU()
        )
        self.frag_attend = nn.Sequential(
            nn.Linear(self.num_heads * net_params['hidden_dim'], net_params['hidden_dim'], bias=True),
            nn.BatchNorm1d(net_params['hidden_dim']),
            nn.ReLU()
        )
        self.motif_attend = nn.Sequential(
            nn.Linear(self.num_heads * net_params['hidden_dim'], net_params['hidden_dim'], bias=True),
            nn.BatchNorm1d(net_params['hidden_dim']),
            nn.ReLU()
        )

        #self.LN = nn.LayerNorm(3 * net_params['hidden_dim'])

        self.linear_predict0 = nn.Sequential(
            nn.Linear(net_params['hidden_dim'], net_params['hidden_dim'], bias=True),
            nn.BatchNorm1d(net_params['hidden_dim']),
            nn.ReLU()
        )

        self.linear_predict1 = nn.Sequential(
            nn.Linear(net_params['hidden_dim'], net_params['hidden_dim'], bias=True),
            nn.BatchNorm1d(net_params['hidden_dim']),
            nn.ELU()
        )

        self.GRUCell = nn.GRUCell(net_params['hidden_dim'], net_params['hidden_dim'])
        #self.linear_predict1 = nn.Sequential(
        #    nn.Dropout(net_params['dropout']),
        #    nn.Linear(3 * net_params['hidden_dim'], net_params['hidden_dim'], bias=True),
        #    nn.BatchNorm1d(net_params['hidden_dim']),
        #)
        self.linear_predict2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(net_params['hidden_dim'], 1, bias=True)
        )


    def reset_parameters(self):
        for origin_layer in self.origin_heads:
            origin_layer.reset_parameters()
        for fragment_layer in self.fragment_heads:
            fragment_layer.reset_parameters()
        for junction_layer in self.junction_heads:
            junction_layer.reset_parameters()
        for layer in self.linear_predict1:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        for layer in self.linear_predict2:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, origin_graph, origin_node, origin_edge, frag_graph, frag_node, frag_edge, motif_graph, motif_node, motif_edge, get_descriptors = False):
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
        origin_node = origin_node.float()
        origin_edge = origin_edge.float()
        origin_node = self.embedding_node_lin(origin_node)
        origin_edge = self.embedding_edge_lin(origin_edge)
        # Origin Layer:
        origin_heads_out = [origin_block(origin_graph, origin_node, origin_edge) for origin_block in self.origin_heads]
        #graph_origin = torch.relu(torch.mean(torch.stack(origin_heads_out), dim=0))
        graph_origin = self.origin_attend(torch.cat(origin_heads_out, dim=-1))

        # Fragments Layer:
        frag_node = frag_node.float()
        frag_edge = frag_edge.float()
        frag_node = self.embedding_node_lin(frag_node)
        frag_edge = self.embedding_edge_lin(frag_edge)
        frag_heads_out = [frag_block(frag_graph, frag_node, frag_edge) for frag_block in self.fragment_heads]
        #graph_motif = torch.mean(torch.stack(frag_heads_out), dim=0)
        graph_motif = self.frag_attend(torch.cat(frag_heads_out, dim=-1))
        motif_graph.ndata['feats'] = graph_motif
        motifs_series = torch.relu(dgl.sum_nodes(motif_graph, 'feats'))
        # Junction Tree Layer:
        #motif_node = motif_node.float()
        motif_edge = motif_edge.float()
        motif_node = self.embedding_frag_lin(motif_node)
        motif_edge = self.embedding_edge_lin(motif_edge)
        motif_node = torch.cat([graph_motif, motif_node], dim=-1)
        junction_graph_heads_out = []
        junction_attention_heads_out = []
        for single_head in self.junction_heads:
            single_head_new_graph, single_head_attention_weight = single_head(motif_graph, motif_node, motif_edge)
            junction_graph_heads_out.append(single_head_new_graph)
            junction_attention_heads_out.append(single_head_attention_weight)

        super_new_graph = torch.relu(torch.mean(torch.stack(junction_graph_heads_out, dim=1), dim=1))
        super_attention_weight = torch.mean(torch.stack(junction_attention_heads_out, dim=1), dim=1)
        #concat_features = torch.cat([graph_origin, super_new_graph, motifs_series], dim=-1)
        concat_features = self.linear_predict0(graph_origin)
        context = self.linear_predict1(super_new_graph)
        #norm_concat_features = self.LN(concat_features)
        #descriptors = self.linear_predict1(concat_features)
        descriptors = self.GRUCell(context, concat_features)
        output = self.linear_predict2(descriptors)
        motif_graph.ndata['attention_weight'] = super_attention_weight
        #return output, motif_graph
        if get_descriptors:
            return output, descriptors
        else:
            return output

    def loss(self, scores, targets):
        loss = nn.MSELoss()(scores, targets)
        return loss



