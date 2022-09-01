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

class Origin_Channel(nn.Module):
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
        self.origin_heads = nn.ModuleList([SingleHeadOriginLayer(net_params) for _ in range(net_params['num_heads'])])
        self.origin_attend = nn.Sequential(
            nn.Linear(net_params['num_heads'] * net_params['hidden_dim'], net_params['hidden_dim'], bias=True),
            nn.BatchNorm1d(net_params['hidden_dim']),
            nn.ReLU()
        )

    def reset_parameters(self):
        for l in self.embedding_node_lin:
            if isinstance(l, nn.Linear):
                l.reset_parameters()
        for l in self.embedding_edge_lin:
            if isinstance(l, nn.Linear):
                l.reset_parameters()
        for l in self.origin_attend:
            if isinstance(l, nn.Linear):
                l.reset_parameters()
        for ol in self.origin_heads:
            ol.reset_parameters()

    def forward(self, origin_graph, origin_node, origin_edge):
        origin_node = self.embedding_node_lin(origin_node.float())
        origin_edge = self.embedding_edge_lin(origin_edge.float())
        origin_heads_out = [origin_block(origin_graph, origin_node, origin_edge) for origin_block in self.origin_heads]
        graph_origin = self.origin_attend(torch.cat(origin_heads_out, dim=-1))
        return graph_origin


class Fragment_Channel(nn.Module):
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
        self.fragment_heads = nn.ModuleList([SingleHeadFragmentLayer(net_params) for _ in range(net_params['num_heads'])])
        self.frag_attend = nn.Sequential(
            nn.Linear(net_params['num_heads'] * net_params['hidden_dim'], net_params['hidden_dim'], bias=True),
            nn.BatchNorm1d(net_params['hidden_dim']),
            nn.ReLU()
        )

    def reset_parameters(self):
        for l in self.embedding_node_lin:
            l.reset_parameters()
        for l in self.embedding_edge_lin:
            l.reset_parameters()
        for l in self.frag_attend:
            l.reset_parameters()
        for ol in self.fragment_heads:
            ol.reset_parameters()

    def forward(self, frag_graph, frag_node, frag_edge):
        frag_node = self.embedding_node_lin(frag_node.float())
        frag_edge = self.embedding_edge_lin(frag_edge.float())
        frag_heads_out = [frag_block(frag_graph, frag_node, frag_edge) for frag_block in self.fragment_heads]
        graph_motif = self.frag_attend(torch.cat(frag_heads_out, dim=-1))
        return graph_motif


class JT_Channel(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.embedding_frag_lin = nn.Sequential(
            nn.Linear(net_params['frag_dim'], net_params['hidden_dim'], bias=True),
            nn.BatchNorm1d(net_params['hidden_dim']),
            nn.LeakyReLU()
        )
        self.embedding_edge_lin = nn.Sequential(
            nn.Linear(net_params['num_bond_type'], net_params['hidden_dim'], bias=True),
            nn.BatchNorm1d(net_params['hidden_dim']),
            nn.LeakyReLU()
        )
        self.junction_heads = nn.ModuleList([SingleHeadJunctionLayer(net_params) for _ in range(net_params['num_heads'])])

    def forward(self, graph_motif, motif_graph, motif_node, motif_edge):
        motif_node = self.embedding_frag_lin(motif_node)
        motif_edge = self.embedding_edge_lin(motif_edge.float())
        motif_node = torch.cat([graph_motif, motif_node], dim=-1)
        junction_graph_heads_out = []
        junction_attention_heads_out = []
        for single_head in self.junction_heads:
            single_head_new_graph, single_head_attention_weight = single_head(motif_graph, motif_node, motif_edge)
            junction_graph_heads_out.append(single_head_new_graph)
            junction_attention_heads_out.append(single_head_attention_weight)
        super_new_graph = torch.relu(torch.mean(torch.stack(junction_graph_heads_out, dim=1), dim=1))
        super_attention_weight = torch.mean(torch.stack(junction_attention_heads_out, dim=1), dim=1)
        return super_new_graph, super_attention_weight


class GCGAT_v4(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.origin_module = Origin_Channel(net_params)
        self.frag_module = Fragment_Channel(net_params)
        self.junction_module = JT_Channel(net_params)
        self.linear_predict1 = nn.Sequential(
            nn.Dropout(net_params['dropout']),
            nn.Linear(net_params['hidden_dim'], net_params['hidden_dim'], bias=True),
            nn.BatchNorm1d(net_params['hidden_dim']),
        )
        self.linear_predict2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(net_params['hidden_dim'], 1, bias=True)
        )
        self.ChannelEmbedding = Mol_AttentiveFP(net_params)
        #self.reset_parameters()

    def reset_parameters(self):
        self.origin_module.reset_parameters()
        self.frag_module.reset_parameters()
        self.junction_module.reset_parameters()
        self.ChannelEmbedding.reset_parameters()
        for layer in self.linear_predict2:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, origin_graph, origin_node, origin_edge, frag_graph, frag_node, frag_edge, motif_graph, motif_node, motif_edge, channel_graph, index, get_descriptors = False, get_attention=False):
        graph_origin = self.origin_module(origin_graph, origin_node, origin_edge)
        graph_motif = self.frag_module(frag_graph, frag_node, frag_edge)
        motif_graph.ndata['feats'] = graph_motif
        motifs_series = torch.relu(dgl.sum_nodes(motif_graph, 'feats'))
        super_new_graph, super_attention_weight = self.junction_module(graph_motif, motif_graph, motif_node, motif_edge)

        concat_features = torch.cat([graph_origin, super_new_graph, motifs_series], dim=0)
        channel_node = concat_features[index]
        descriptors, super_node_attention = self.ChannelEmbedding(channel_graph, channel_node)

        output = self.linear_predict2(self.linear_predict1(descriptors))
        if get_attention:
            motif_graph.ndata['attention_weight'] = super_attention_weight
            attention_list_array = []
            for g in dgl.unbatch(motif_graph):
                attention_list_array.append(g.ndata['attention_weight'].detach().to(device='cpu').numpy())
            return output, attention_list_array
        if get_descriptors:
            return output, descriptors
        else:
            return output

    def loss(self, scores, targets):
        loss = nn.MSELoss()(scores, targets)
        return loss



