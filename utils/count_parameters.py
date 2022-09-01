# -*- coding: utf-8 -*-
# @Time    : 2022/4/5 12:03
# @Author  : FAN FAN
# @Site    : 
# @File    : count_parameters.py
# @Software: PyCharm
from prettytable import PrettyTable


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