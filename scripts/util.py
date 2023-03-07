# -*- coding: utf-8 -*-
import torch
device = torch.device("cuda")

def editDistance(x, y, pad_token):
    """
    x is a tensor of shape (N, L)
    y is a tensor of shape (N, L')
    """
    N, x_len = x.shape
    y_len = y.size(1)
    v0 = torch.arange(0, x_len+1, device=device).repeat(N, 1)

    for i in range(y_len):
        v1 = v0 + 1
        deletionCost = v0[:, :-1] + 1
        substitutionCost = v0[:, :-1] + (x == y[:, i, None]).int()
        v1[:, 1:] = torch.minimum(v1[:, 1:], substitutionCost)
        v1[:, 1:] = torch.minimum(v1[:, 1:], deletionCost)
        v0 = v1

    return v0[:, -1]


    