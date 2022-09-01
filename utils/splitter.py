# -*- coding: utf-8 -*-
# @Time    : 2022/3/23 02:25
# @Author  : FAN FAN
# @Site    : 
# @File    : splitter.py
# @Software: PyCharm
import torch
import numpy as np
from torch.utils import data
from torch.utils.data import Subset
from collections import defaultdict

class Splitter(data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.length = len(dataset)

    def Random_Splitter(self, seed=100, frac_train=0.7, frac_val=0.15):
        # Return 4 objects: Train Dataset, Val Dataset, Test Dataset, Complete Dataset
        rand_state = np.random.RandomState(int(seed))
        indices = [*range(self.length)]
        rand_state.shuffle(indices)

        num_train = int(frac_train * self.length)
        num_val = int(frac_val * self.length)

        train_indices = indices[:num_train]
        val_indices = indices[num_train:num_train + num_val]
        test_indices = indices[num_train + num_val:]

        train_dataset = Subset(self.dataset, train_indices)
        val_dataset = Subset(self.dataset, val_indices)
        test_dataset = Subset(self.dataset, test_indices)
        return train_dataset, val_dataset, test_dataset, self.dataset

    def Stratified_Target_Splitter(self, bin_size=20, seed=100, frac_train=0.7, frac_val=0.15):
        '''Splits the dataset by stratification
        Step.1. Sort the molecules based on their true values following a given order.
        Step.2. Move the sorted molecules to the pre-defined bins by fixed intervals.
        Step.3. Take buckets of datapoints repeatly from these bins to augment the training, validation and test subsets.
        '''
        rand_state = np.random.RandomState(int(seed))
        bin_indices = [*range(bin_size)]
        rand_state.shuffle(bin_indices)

        sorted_indices = np.argsort(self.dataset.targets_list)
        train_indices, val_indices, test_indices = [], [], []
        num_train_bin = int(frac_train * bin_size)
        num_val_bin = int(frac_val * bin_size)
        train_bin_indices = bin_indices[: num_train_bin]
        val_bin_indices = bin_indices[num_train_bin: num_train_bin + num_val_bin]
        test_bin_indices = bin_indices[num_train_bin + num_val_bin: ]

        while sorted_indices.shape[0] >= bin_size:
            current_bin, sorted_indices = np.split(sorted_indices, [bin_size])
            train_indices.extend(current_bin[train_bin_indices].tolist())
            val_indices.extend(current_bin[val_bin_indices].tolist())
            test_indices.extend(current_bin[test_bin_indices].tolist())

        # Then put rest samples from the last bin into subsets.
        bin_size = sorted_indices.shape[0]
        num_train_bin = int(frac_train * bin_size)
        num_val_bin = int(frac_val * bin_size)
        bin_indices = [*range(bin_size)]
        rand_state.shuffle(bin_indices)
        train_indices.extend(sorted_indices[bin_indices[: num_train_bin]].tolist())
        val_indices.extend(sorted_indices[bin_indices[num_train_bin: num_train_bin + num_val_bin]].tolist())
        test_indices.extend(sorted_indices[bin_indices[num_train_bin + num_val_bin: ]].tolist())

        train_dataset = Subset(self.dataset, train_indices)
        val_dataset = Subset(self.dataset, val_indices)
        test_dataset = Subset(self.dataset, test_indices)

        return train_dataset, val_dataset, test_dataset, self.dataset

    def Stratified_Family_Splitter(self, bin_size=20, seed=100, frac_train=0.7, frac_val=0.15):
        '''Splits the dataset by stratification
        Step.1. Sort the molecules based on their true values following a given order.
        Step.2. Move the sorted molecules to the pre-defined bins by fixed intervals.
        Step.3. Take buckets of datapoints repeatly from these bins to augment the training, validation and test subsets.
        '''
        rand_state = np.random.RandomState(int(seed))
        family_list = self.dataset.family_list
        families, counts = np.unique(self.dataset.family_list, return_counts=True)
        #num_families = l
        print('Num of families:', len(families))
        print('Maximum size:', np.max(counts))
        print('Minimum size:', np.min(counts))
        family_bin_index = defaultdict(list)
        for idx, item in enumerate(family_list):
            family_bin_index[item].append(idx)

        train_indices, val_indices, test_indices = [], [], []
        for key, value in family_bin_index.items():
            bin_size = len(value)
            bin_indices = [*range(bin_size)]
            rand_state.shuffle(bin_indices)
            num_train_bin = int(frac_train * bin_size)
            num_val_bin = int(frac_val * bin_size)
            num_test_bin = int((1 - frac_train - frac_val) * bin_size)
            if num_val_bin != 0 and num_test_bin != 0:
                train_indices.extend(value[bin_indices[: num_train_bin]].tolist())
                val_indices.extend(value[bin_indices[num_train_bin: num_train_bin + num_val_bin]].tolist())
                test_indices.extend(value[bin_indices[num_train_bin + num_val_bin: ]].tolist())


        train_dataset = Subset(self.dataset, train_indices)
        val_dataset = Subset(self.dataset, val_indices)
        test_dataset = Subset(self.dataset, test_indices)

        return train_dataset, val_dataset, test_dataset, self.dataset