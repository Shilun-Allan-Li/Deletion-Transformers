# -*- coding: utf-8 -*-
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F

device = torch.device("cuda")
"""
The input and output of the channel are tensor of shape (batch, code_length) and Long type on cuda
The output should also contain the lengths (even if no deletion occurred)
"""


def deletionChannel(x, p, pad_token):
    # x is of shape (N, codeword length)
    deletion_mask = torch.rand(x.shape, device=device) > p
    deleted_samples_seq = [x[i][deletion_mask[i]] for i in range(deletion_mask.size(0))]
    src_len = torch.tensor([len(deleted_samples_seq[i]) for i in range(deletion_mask.size(0))])
    deleted_samples = nn.utils.rnn.pad_sequence(deleted_samples_seq, padding_value=pad_token)
    return deleted_samples, src_len


def BSCChannel(x, p):
    # x is of shape (N, codeword length)
    mask = torch.rand(x.shape, device=device) < p
    return torch.logical_xor(x, mask).long(), torch.tensor([x.size(1)]*x.size(0), device=device)