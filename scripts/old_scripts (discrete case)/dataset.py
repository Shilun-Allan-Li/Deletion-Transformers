# -*- coding: utf-8 -*-
import torch
import random
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class MarkovCodeDataset(Dataset):
    def __init__(self, dataset_size, length, deletion_p, transition_p):
        self.dataset_size = dataset_size
        self.length = length
        self.deletion_p = deletion_p
        self.transition_p = transition_p

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        transitions = torch.rand(self.length) < self.transition_p
        x = torch.zeros(self.length, dtype=float)
        x[0] = 1 if torch.rand(1) < 0.5 else -1
        for i in range(1, self.length):
            x[i] = -x[i-1] if transitions[i] else x[i-1]
        mask = torch.rand(self.length) > self.deletion_p
        y = torch.masked_select(x, mask)
        y_len = len(y)
        y = F.pad(y, (0, self.length-y_len), 'constant', 0.0)
        y_mask = torch.ones(y_len)
        y_mask = F.pad(y_mask, (0, self.length-y_len), 'constant', 0.0)
        return x, y, mask, y_mask

def data_loader(dataset_size, length, deletion_p, transition_p, batch_size):
    dset = MarkovCodeDataset(
    dataset_size=dataset_size,
    length=length,
    deletion_p=deletion_p,
    transition_p=transition_p)

    loader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=False)
    return dset, loader