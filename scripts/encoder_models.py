# -*- coding: utf-8 -*-
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class LSTMEncoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.linear = nn.Conv1d(args.message_length, args.code_length, kernel_size=1, padding=0)
        self.lstm = nn.LSTM(args.alphabet_size, args.alphabet_size, batch_first=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x has size (batch, message length, alphabet size)
        hidden = self.linear(x)
        logits, _ = self.lstm(hidden)
        # output size is (batch, code length, alphabet size)
        output = self.softmax(logits)
        return output


class RandomSystematicLinearEncoding(nn.Module):
    def __init__(self, args):
        super().__init__()
        matrix = torch.rand((args.message_length, args.code_length - args.message_length)) > 0.5
        self.register_buffer('matrix', matrix)

    def forward(self, x):
        return torch.cat(x, x @ self.matrix, dim=-1)


class RandomLinearEncoding(nn.Module):
    def __init__(self, args):
        super().__init__()
        matrix = torch.rand((args.message_length, args.code_length)) > 0.5
        self.register_buffer('matrix', matrix)

    def forward(self, x):
        return x @ self.matrix

