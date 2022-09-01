# -*- coding: utf-8 -*-
# @Time    : 2022/6/9 13:27
# @Author  : FAN FAN
# @Site    : 
# @File    : count_parameters.py
# @Software: PyCharm
from data.model_library import load_model, load_params
from prettytable import PrettyTable
from networks.DMPNN import DMPNNNet as DMPNN
from networks.MPNN import MPNNNet as MPNN
from networks.AttentiveFP import AttentiveFPNet as AFP
from networks.FraGAT import NewFraGATNet as FraGAT
from networks.AGC import AGCNet as AGC

def count_parameters(model, verbose=0):
    """ Count the number of learnable-parameters in model
    Parameters
    ----------
    model : pytorch.object
        Model for training.
    verbose : bool
        Whether to show table. Default to 0.

    Returns
    ----------
    total_params : int
        Number of learnable-parameters
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    if verbose == 1:
        print(table)
        print(f"Total Trainable Params: {total_params}")
    return total_params


path = 'library/Ensembles/BMP_FraGAT_0/Ensemble_0_BMP_FraGAT_settings'
params, net_params = load_params(path, idx=73)
model = FraGAT(net_params).to(device='cuda')
n_param = count_parameters(model)
print(n_param)