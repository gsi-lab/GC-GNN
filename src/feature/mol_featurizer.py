# -*- coding: utf-8 -*-
# @Time    : 2022/3/23 14:31
# @Author  : FAN FAN, chn.tz.fanfan@outlook.com
# @Site    : 
# @File    : mol_featurizer.py
# @Software: PyCharm

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors
import numpy as np
import torch


def global_feature(mol):
    num_atom = mol.GetNumAtoms()
    num_bond = mol.GetNumBonds()
    mole_weight = Descriptors.MolWt(mol)
    # Number of AliphaticRings existing
    num_aliph_ring = rdMolDescriptors.CalcNumAliphaticRings(mol)
    # Number of AromaticRings existing
    num_aroma_ring = rdMolDescriptors.CalcNumAromaticRings(mol)
    return np.asarray([num_atom,
                       mole_weight,
                       num_aliph_ring,
                       num_aroma_ring])

# commonly-used mol-level global featurizer
def classic_mol_featurizer(mol):
    num_atom = mol.GetNumAtoms()
    global_feats = np.asarray([global_feature(mol) for _ in range(num_atom)])
    return torch.tensor(global_feats)
