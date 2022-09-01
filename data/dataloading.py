"""
======================================================================================================================
0. Introduction
This script aims to load data and generate map-style dataset for further implementation.
    -Import raw dataset.
    -Import extra dataset.

By FAN FAN (s192217@dtu.dk)
======================================================================================================================
1. Load Package
Loading the necessary package:
torch, numpy, pandas
======================================================================================================================
2. Steps
======================================================================================================================
"""
import os
import platform

import numpy as np
import pandas as pd

from .scaler import Standardization
from utils.featurizer import get_canonical_smiles

# 'Si', 'Al', 'Mg', 'Zn', 'Cr', 'Na', 'K', 'Pb'
ATOM_REMOVE = ['Sn', 'As', 'Ti', 'Ca', 'Fe']
def import_dataset(params):
    DATASET_NAME = params['Dataset']
    path = os.path.join(
            './/datasets',
            DATASET_NAME + '.csv')
    data_from = os.path.realpath(path)
    # Lad the data
    if DATASET_NAME == 'AF':
        df = pd.read_csv(data_from, sep=';', encoding='windows=1250')
    else:
        df = pd.read_csv(data_from, sep=',', encoding='windows=1250')
    # drop non-necessary columns
    if DATASET_NAME not in ['ESOL', 'HSOL_D', 'HSOL_H', 'HSOL_P', 'HSOL_D2', 'HSOL_H2', 'HSOL_P2', 'BMP', 'HFOR_capec', 'HFOR_amol']:
        df = df.drop(df[df['Family'] == 'OTHER INORGANICS'].index).drop(df[df['Family'] == 'Other Inorganics'].index)
        df.drop(labels=['CNAM', 'INAM', 'PropertyID', 'Data_Type', 'Acceptance', 'Error_Source'], axis=1,
                inplace=True)
        df.dropna(axis='rows', how='any', inplace=True)
    # remove heavy metal atoms:
    for i in ATOM_REMOVE:
        df.drop(df[df.SMILES.str.contains(i)].index, inplace=True)
        df.reset_index(drop=True, inplace=True)

    # rename column
    # df.rename(columns={'Const_Value': DATASET_NAME}, inplace=True)
    # substitute 'Unknown' entries in the error section with nan
    if DATASET_NAME not in ['ESOL', 'AF', 'HSOL_D', 'HSOL_H', 'HSOL_P', 'HSOL_D2', 'HSOL_H2', 'HSOL_P2', 'BMP', 'HFOR_capec', 'HFOR_amol']:
        df.loc[df.Error == 'Unknown', 'Error'] = '< nan%'
        # splitting
        df['Error'] = df['Error'].str.split('< ', n=1, expand=True)[1]
        df['Error'] = df['Error'].str.split('%', n=0, expand=True)[0]
        # Converting the strings and nan to numerics
        df['Error'] = pd.to_numeric(df['Error'], errors='coerce')
        # The entries with nan for errors are assigned the average error value
        df.loc[pd.isna(df.Error), 'Error'] = df['Error'].mean(skipna=True)
    if 'Name ' in df.columns:
        df.rename(columns={'Name ':'NAMES'}, inplace=True)

    # keep error for further comparison
    Scaling = Standardization(df['Const_Value'])
    df['Const_Value'] = Scaling.Scaler(df['Const_Value'])
    df['SMILES'] = get_canonical_smiles(df['SMILES'])
    df.rename(columns={'Name':'NAMES'}, inplace=True)
    return df, Scaling


#def load_data(params, dataset):
#    smiles_list, label_list, names_list, family_list, Scaling = import_dataset(params)
#    return MolDataset(params, smiles_list=smiles_list, targets_list=label_list, names_list=names_list,
#                          family_list=family_list, scaling=Scaling)


def import_extra_dataset(params):
    DATASET_NAME = params['Dataset']
    path = os.path.join(
            './/datasets',
            DATASET_NAME + '.csv')
    data_from = os.path.realpath(path)
    # Lad the data
    if DATASET_NAME == 'AF':
        df = pd.read_csv(data_from, sep=';', encoding='windows=1250')
    else:
        df = pd.read_csv(data_from, sep=',', encoding='windows=1250')

    df['SMILES'] = get_canonical_smiles(df['SMILES'])
    df.rename(columns={'Name': 'NAMES'}, inplace=True)
    return df


#def load_extra_data(params, scaling):
#    smiles_list, label_list, names_list, family_list = import_extra_dataset(params, scaling)
#    return MolDataset(params, smiles_list=smiles_list, targets_list=label_list, names_list=names_list,
#                      family_list=family_list, scaling=scaling)

