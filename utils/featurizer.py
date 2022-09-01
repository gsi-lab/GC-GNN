# -*- coding: utf-8 -*-
# @Time    : 2022/3/23 02:58
# @Author  : FAN FAN
# @Site    : 
# @File    : featurizer.py
# @Software: PyCharm
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.EState import EState
from rdkit.Chem.rdchem import BondType

def get_canonical_smiles(smiles):
    smi_list = []
    for s in smiles:
        try:
            smi_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(s)))
        except:
            print('Failed to generate the canonical smiles from ', s, ' . Please check the inputs.')

    return smi_list


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: float(x == s), allowable_set))


def chirality_type(x, allowable_set):
    if x.HasProp('_CIPCode'):
        return one_of_k_encoding_unk(str(x.GetProp('_CIPCode')), allowable_set)
    else:
        return [0, 0]


def global_feature(mol):
    num_atom = mol.GetNumAtoms()
    num_bond = mol.GetNumBonds()
    mole_weight = Descriptors.MolWt(mol)
    # Number of AliphaticRings exist
    num_aliph_ring = rdMolDescriptors.CalcNumAliphaticRings(mol)
    # Number of AromaticRings exist
    num_aroma_ring = rdMolDescriptors.CalcNumAromaticRings(mol)
    return np.asarray([num_atom,
                       mole_weight,
                       num_aliph_ring,
                       num_aroma_ring])


def edge_feature(bond):
    bt = bond.GetBondType()
    stereo = bond.GetStereo()
    fbond = np.asarray([
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
        stereo == Chem.rdchem.BondStereo.STEREONONE,
        stereo == Chem.rdchem.BondStereo.STEREOANY,
        stereo == Chem.rdchem.BondStereo.STEREOZ,
        stereo == Chem.rdchem.BondStereo.STEREOE,
        stereo == Chem.rdchem.BondStereo.STEREOCIS,
        stereo == Chem.rdchem.BondStereo.STEREOTRANS
    ])
    return fbond

def edge_topol_distance(mol):
    distance_map = AllChem.GetDistanceMatrix(mol)
    return distance_map


def node_feature(atom):
    ATOM_VOCAB = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I']
    # 'CHI_TETRAHEDRAL_CW': clockwise, 'CHI_TETRAHEDRAL_CCW': anti-clockwise
    return np.asarray(
        one_of_k_encoding_unk(atom.GetSymbol(), ATOM_VOCAB) +
        one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
        one_of_k_encoding_unk(str(atom.GetHybridization()), ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2']) +
        [atom.GetIsAromatic()] +
        [atom.HasProp('_ChiralityPossible')] +
        chirality_type(atom, ['R', 'S']) +
        one_of_k_encoding_unk(str(atom.GetChiralTag()), ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW']) +
        [atom.GetFormalCharge()]
    )
'''
mol = Chem.MolFromSmiles(smi[0])
mol2 = Chem.MolFromSmiles(smi2[0])
mol_size = mol.GetNumAtoms()
chiral = np.zeros((mol_size, 39))
i = 0
for atom in mol.GetAtoms():
    chiral[i, :] = node_feature(atom)
    #print(atom.GetChiralTag())
    i += 1
chiral2 = np.zeros((mol_size, 39))
i = 0
for atom in mol2.GetAtoms():
    chiral2[i, :] = node_feature(atom)
    #print(atom.GetChiralTag())
    i += 1

temp = chiral - chiral2
idx = np.where(temp != 0)
print(idx)
'''

def node_feature_mol_level(mol):
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
