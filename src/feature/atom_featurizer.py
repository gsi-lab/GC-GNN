# -*- coding: utf-8 -*-
# @Time    : 2022/3/23 02:56
# @Author  : FAN FAN, chn.tz.fanfan@outlook.com
# @Site    :
# @File    : atom_featurizer.py
# @Software: PyCharm

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.EState import EState
import torch

ATOM_VOCAB = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I']


def one_of_k_encoding_unk(x, allowable_set):
    # one-hot features converter
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: float(x == s), allowable_set))


def chirality_type(x, allowable_set):
    # atom's chirality type
    if x.HasProp('_CIPCode'):
        return one_of_k_encoding_unk(str(x.GetProp('_CIPCode')), allowable_set)
    else:
        return [0, 0]


def node_feature(atom):
    return np.asarray(
        one_of_k_encoding_unk(atom.GetSymbol(), ATOM_VOCAB) +
        one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
        one_of_k_encoding_unk(str(atom.GetHybridization()), ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2']) +
        [atom.GetIsAromatic()] +
        [atom.HasProp('_ChiralityPossible')] +
        chirality_type(atom, ['R', 'S']) +
        [atom.GetFormalCharge()]
    )


#one_of_k_encoding_unk(str(atom.GetChiralTag()), ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW']) +
def node_feature_mol_level(mol):
    # These features need computation on the level of overall molecule.
    # GasteigerCharges & MMFF94 Charges
    AllChem.ComputeGasteigerCharges(mol)
    # fps = AllChem.MMFFGetMoleculeProperties(mol)
    charges = []
    for atom in mol.GetAtoms():
        charges.append(float(atom.GetProp('_GasteigerCharge')))

    # Crippen Contributions to LogP and Molar Refractivity
    crippen = rdMolDescriptors._CalcCrippenContribs(mol)
    mol_log = []
    mol_mr = []
    for x, y in crippen:
        mol_log.append(x)
        mol_mr.append(y)

    # Labute Approximate Surface Area Contribution
    asa = rdMolDescriptors._CalcLabuteASAContribs(mol)
    lasa = [x for x in asa[0]]

    # Total Polar Surface Area (TPSA) contribution
    tpsa = rdMolDescriptors._CalcTPSAContribs(mol)

    # Electrotopological State
    estate = EState.EStateIndices(mol)

    # x-y-z coordinates
    # coord = con.GetPositions()
    # x_coord = coord[:, 0]
    # y_coord = coord[:, 1]
    # z_coord = coord[:, 2]
    return np.column_stack([charges] +
                           [mol_log] +
                           [mol_mr] +
                           [lasa] +
                           [list(tpsa)] +
                           [estate.tolist()])


# commonly-used atom featurizer
def classic_atom_featurizer(mol):
    num_atoms = mol.GetNumAtoms()
    atom_list = [mol.GetAtomWithIdx(i) for i in range(num_atoms)]
    node_feats = np.asarray([node_feature(atom) for atom in atom_list])
    #print(node_feats.shape)
    return torch.tensor(node_feats)

# featurizer added with corey's mol-level features, details in
def extended_atom_featurizer(mol):
    num_atoms = mol.GetNumAtoms()
    atom_list = [mol.GetAtomWithIdx(i) for i in range(num_atoms)]
    node_feats = np.asarray([node_feature(atom) for atom in atom_list])
    x_node_feats = node_feature_mol_level(mol)
    ext_node_feats = np.concatenate((node_feats, x_node_feats), axis=1)
    return torch.tensor(ext_node_feats)