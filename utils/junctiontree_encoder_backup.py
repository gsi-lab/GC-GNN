# -*- coding: utf-8 -*-
# @Time    : 2022/4/3 11:13
# @Author  : FAN FAN
# @Site    : 
# @File    : junctiontree_encoder.py
# @Software: PyCharm
import numpy as np
from rdkit import Chem
import os
import pandas as pd
import dgl
import torch


class JT_SubGraph(object):
    def __init__(self):
        #self.smiles_list = smiles_list
        path = os.path.join('.//datasets', 'MG_reference' + '.csv')
        data_from = os.path.realpath(path)
        df = pd.read_csv(data_from)
        pattern = np.array([df['First-Order Group'], df['SMARTs'], df['Priority']])
        self.sorted_pattern = pattern[:, np.argsort(pattern[2, :])]
        self.frag_name_list = list(dict.fromkeys(self.sorted_pattern[0, :]))
        self.frag_dim = len(self.frag_name_list)
        #self.max_mol_size = np.max([Chem.MolFromSmiles(smiles).GetNumAtoms() for _, smiles in enumerate(self.smiles_list)])

    def fragmentation(self, graph, mol):
        """Fragmentation
        Parameters
        ----------
        graph : DGLGraph
            DGLGraph for a batch of graphs.
        index : int
            Index of molecule in dataset.
        """
        pat_list = []
        mol_size = mol.GetNumAtoms()
        for patt in self.sorted_pattern[1, :]:
            pat = Chem.MolFromSmarts(patt)
            pat_list.append(list(mol.GetSubstructMatches(pat)))

        num_atoms = mol.GetNumAtoms()
        atom_idx_list = [i for i in range(num_atoms)]
        # hit_ats: dictionary of each fragments and their atom idx, values in np.array
        hit_ats = {}
        prior_set = set()
        k = 0

        for idx, key in enumerate(self.sorted_pattern[0, :]):
            frags = pat_list[idx]
            if frags:
                # print(frags)
                # check overlapping on same subgroups: e.x. trimethylamine 2*CH3, 1*CH3N
                for i, item in enumerate(frags):
                    item_set = set(item)
                    new_frags = frags[:i] + frags[i + 1:]
                    left_set = set(sum(new_frags, ()))
                    if not item_set.isdisjoint(left_set):
                        frags = new_frags
                for _, frag in enumerate(frags):
                    # cur_idx = set(list(itertools.chain(*pat_list[idx])))
                    frag_set = set(frag)  # tuple -> set
                    if prior_set.isdisjoint(frag_set):
                        ats = frag_set
                    else:
                        ats = {}
                    if ats:
                        adjacency_origin = Chem.rdmolops.GetAdjacencyMatrix(mol)[np.newaxis, :, :]
                        if k == 0:
                            adj_mask = adjacency_origin
                            atom_mask = np.zeros((1, mol_size))
                            frag_features = np.asarray(list(map(lambda s: float(key == s), self.frag_name_list)))
                        else:
                            # adjacency_origin = Chem.rdmolops.GetAdjacencyMatrix(m)[np.newaxis, :, :]
                            adj_mask = np.vstack((adj_mask, adjacency_origin))
                            atom_mask = np.vstack((atom_mask, np.zeros((1, mol_size))))
                            frag_features = np.vstack((frag_features,
                                                       np.asarray(
                                                           list(map(lambda s: float(key == s), self.frag_name_list)))))
                        if key not in hit_ats.keys():
                            hit_ats[key] = np.asarray(list(ats))
                        else:
                            hit_ats[key] = np.vstack((hit_ats[key], np.asarray(list(ats))))
                        ignores = list(set(atom_idx_list) - set(ats))

                        adj_mask[k, ignores, :] = 0
                        adj_mask[k, :, ignores] = 0
                        atom_mask[k, list(ats)] = 1
                        k += 1
                        prior_set.update(ats)

        # unknown fragments:
        unknown_ats = list(set(atom_idx_list) - prior_set)
        if len(unknown_ats) > 0:
            for i, at in enumerate(unknown_ats):
                if k == 0:
                    adj_mask = adjacency_origin
                    atom_mask = np.zeros((1, mol_size))
                else:
                    # adjacency_origin = Chem.rdmolops.GetAdjacencyMatrix(m)[np.newaxis, :, :]
                    adj_mask = np.vstack((adj_mask, adjacency_origin))
                    atom_mask = np.vstack((atom_mask, np.zeros((1, mol_size))))
                if 'unknown' not in hit_ats.keys():
                    hit_ats['unknown'] = np.asarray(at)
                else:
                    hit_ats['unknown'] = np.vstack((hit_ats['unknown'], np.asarray(at)))
                ignores = list(set(atom_idx_list) - set([at]))
                # print(prior_idx)
                adj_mask[k, ignores, :] = 0
                adj_mask[k, :, ignores] = 0
                atom_mask[k, at] = 1
                frag_features = np.vstack(
                    (frag_features, np.asarray(list(map(lambda s: float('unknown' == s), self.frag_name_list)))))
                k += 1
        # adj_mask: size: [num of fragments, max num of atoms, max num of atoms]
        # atom_mask: size: [num of fragments, max num of atoms]
        # adjacency_fragments: size: [max num of atoms, max num of atoms]
        adjacency_fragments = adj_mask.sum(axis=0)
        idx1, idx2 = (adjacency_origin.squeeze(0) - adjacency_fragments).nonzero()

        # idx_tuples: list of tuples, idx of begin&end atoms on each new edge
        idx_tuples = list(zip(idx1.tolist(), idx2.tolist()))
        # remove reverse edges
        # idx_tuples = list(set([tuple(sorted(item)) for item in idx_tuples]))
        rm_edge_ids_list = []
        for i, item in enumerate(idx_tuples):
            try:
                rm_edge_ids = graph.edge_ids(item[0], item[1])
            except:
                #rm_edge_ids = graph.edge_ids(item[1], item[0])
                continue
            rm_edge_ids_list.append(rm_edge_ids)

        frag_graph = dgl.remove_edges(graph, rm_edge_ids_list)

        num_motifs = atom_mask.shape[0]
        motif_graph = dgl.DGLGraph()
        motif_graph.add_nodes(num_motifs)

        adjacency_motifs, idx_tuples, motif_graph = self.build_adjacency_motifs(atom_mask, idx_tuples, motif_graph)
        #idx_tuples = list(set([tuple(sorted(item)) for item in idx_tuples]))

        if frag_features.ndim == 1:
            frag_features = frag_features.reshape(-1, 1).transpose()

        motif_graph.ndata['feat'] = torch.Tensor(frag_features)
        motif_graph.ndata['atom_mask'] = torch.Tensor(atom_mask)

        edge_features = graph.edata['feat']
        add_edge_feats_ids_list = []
        for i, item in enumerate(idx_tuples):
            try:
                add_edge_feats_ids = graph.edge_ids(item[0], item[1])
            except:
                #add_edge_feats_ids = graph.edge_ids(item[1], item[0])
                continue
            add_edge_feats_ids_list.append(add_edge_feats_ids)
        motif_edge_features = edge_features[add_edge_feats_ids_list, :]
        try:
            motif_graph.edata['feat'] = motif_edge_features

            frag_graph_list = self.rebuild_frag_graph(frag_graph, motif_graph)
            motif_graph.ndata.pop('atom_mask')
            return frag_graph_list, motif_graph
        except:
            frag_graph_list = self.rebuild_frag_graph(frag_graph, motif_graph)
            return frag_graph_list, None

    def atom_locate_frag(self, atom_mask, atom):
        return atom_mask[:, atom].tolist().index(1)

    def frag_locate_atom(self, atom_mask, frag):
        return atom_mask[frag, :].nonzero()[0].tolist()

    def build_adjacency_motifs(self, atom_mask, idx_tuples, motif_graph):
        k = atom_mask.shape[0]
        duplicate_bond = []
        adjacency_motifs = np.zeros((k, k)).astype(int)
        motif_edge_begin = list(map(lambda x: self.atom_locate_frag(atom_mask, x[0]), idx_tuples))
        motif_edge_end = list(map(lambda x: self.atom_locate_frag(atom_mask, x[1]), idx_tuples))
        #adjacency_motifs[new_edge_begin, new_edge_end] = 1
        # eliminate duplicate bond in triangle substructure
        for idx1, idx2 in zip(motif_edge_begin, motif_edge_end):
            if adjacency_motifs[idx1, idx2] == 0:
                adjacency_motifs[idx1, idx2] = 1
                motif_graph.add_edges(idx1, idx2)
            else:
                rm_1 = self.frag_locate_atom(atom_mask, idx1)
                rm_2 = self.frag_locate_atom(atom_mask, idx2)
                if isinstance(rm_1, int):
                    rm_1 = [rm_1]
                if isinstance(rm_2, int):
                    rm_2 = [rm_2]
                for i in rm_1:
                    for j in rm_2:
                        duplicate_bond.extend([tup for tup in idx_tuples if tup == (i, j)])
        if duplicate_bond:
            idx_tuples.remove(duplicate_bond[0])
            idx_tuples.remove(duplicate_bond[2])
        return adjacency_motifs, idx_tuples, motif_graph

    def rebuild_frag_graph(self, frag_graph, motif_graph):
        num_motifs = motif_graph.num_nodes()
        frag_graph_list = []
        for idx_motif in range(num_motifs):
            #new_frag_graph = dgl.DGLGraph()
            coord = motif_graph.nodes[idx_motif].data['atom_mask'].nonzero()
            idx_list = []
            for idx_node in coord:
                idx_list.append(idx_node[1])
            new_frag_graph = dgl.node_subgraph(frag_graph, idx_list)
            frag_graph_list.append(new_frag_graph)
        return frag_graph_list
