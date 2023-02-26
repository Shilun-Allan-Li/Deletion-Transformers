# -*- coding: utf-8 -*-
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F

device = torch.device("cuda")
"""
The input and output of the channel are tensor of shape (batch, code_length, alphabet_size) and float type on cuda
The output should also contain the lengths (even if no deletion occurred)
"""


def deletionChannel(x, p, pad_token, vocab_size):
    # x is of shape (N, codeword length)
    N, code_length = x.shape[:2]
    deletion_mask = torch.rand((N, code_length), device=device) > p
    deleted_samples_seq = [x[i][deletion_mask[i]] for i in range(N)]
    src_len = torch.tensor([len(deleted_samples_seq[i]) for i in range(N)])
    deleted_samples = F.one_hot(torch.tensor([[pad_token]*code_length]*N, device=device), vocab_size).float()
    for i in range(N):
        deleted_samples[i, :src_len[i]] = deleted_samples_seq[i]
    # deleted_samples = nn.utils.rnn.pad_sequence(deleted_samples_seq, padding_value=pad_token, batch_first=True)
    return deleted_samples, src_len


def BSCChannel(x, p):
    # x is of shape (N, codeword length)
    mask = torch.rand(x.shape, device=device) < p
    return torch.logical_xor(x, mask).long(), torch.tensor([x.size(1)]*x.size(0), device=device)

def AWGN(x, SNR):
    sigma2 = 10**(-SNR/10)
    noise = torch.randn(x.shape, dtype=float, device=device)
    return x + noise, torch.tensor([x.size(1)]*x.size(0), device=device)