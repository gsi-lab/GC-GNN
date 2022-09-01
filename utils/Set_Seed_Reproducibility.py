"""
======================================================================================================================
0. Introduction
This script aims to set the reproducibility seed to repeat the outputs.

By FAN FAN (s192217@dtu.dk)
======================================================================================================================
1. Load Package
Loading the necessary package:
numpy
======================================================================================================================
2. Steps
======================================================================================================================
"""
import random
import numpy as np
import dgl
import torch
import os


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    #os.environ['OMP_NUM_THREADS'] = '1'
    #os.environ['MKL_NUM_THREADS'] = '1'
    #torch.set_num_threads(1)
    #np.random.seed(seed) # turn it off during optimization
    random.seed(seed)
    torch.manual_seed(seed)  # annote this line when ensembling
    dgl.random.seed(seed)
    dgl.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        #torch.backends.cudnn.deterministic = True # Faster than the command below and may not fix results
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False