# -*- coding: utf-8 -*-
# @Time    : 2022/3/23 14:05
# @Author  : FAN FAN
# @Site    : 
# @File    : mol2graph.py
# @Software: PyCharm
import torch
import dgl

try:
    from rdkit import Chem
    from rdkit.Chem import rdmolfiles, rdmolops
except ImportError:
    pass

def smiles_2_bigraph(smiles, node_featurizer, edge_featurizer, global_featurizer=None):
    '''
    :param smiles:
    :return:
    '''
    mol = Chem.MolFromSmiles(smiles)
    return mol_2_bigraph(mol, node_featurizer, edge_featurizer, global_featurizer)

def mol_2_bigraph(mol, node_featurizer, edge_featurizer, global_featurizer=None):
    if mol is None:
        print('Invalid molecule!!!')
        return None
    g = bigraph_constructor(mol)
    if node_featurizer is not None:
        g.ndata['feat'] = node_featurizer(mol)
    if edge_featurizer is not None:
        g.edata['feat'] = edge_featurizer(mol)
    if global_featurizer is not None:
        g.ndata['global_feat'] = global_featurizer(mol)
    return g


def graph_2_frag(smiles, origin_graph, JT_subgraph):
    mol = Chem.MolFromSmiles(smiles)
    frag_graph_list, motif_graph, atom_mask, frag_flag = frag_constructor(JT_subgraph, origin_graph, mol)
    return frag_graph_list, motif_graph, atom_mask, frag_flag

def bigraph_constructor(mol):
    # Create empty graph
    g = dgl.DGLGraph()
    # Add nodes
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)
    # Add edges
    src_list = []
    dst_list = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src_list.extend([u, v])
        dst_list.extend([v, u])


    g.add_edges(src_list, dst_list)
    return g

def frag_constructor(JT_subgraph, graph, mol):
    frag_graph_list, motif_graph, atom_mask, frag_flag = JT_subgraph.fragmentation(graph, mol)

    return frag_graph_list, motif_graph, atom_mask, frag_flag


def create_channels():
    # Create empty Graph
    g = dgl.DGLGraph()
    # Add nodes
    g.add_nodes(3)
    g.add_edges([0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1])
    return g

