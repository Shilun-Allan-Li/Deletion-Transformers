# -*- coding: utf-8 -*-
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F

"""
The input to the encoder is tensor of shape (batch, message_length) and Long type on cuda
The output of the encoder should be a tensor of shape (batch, code_length)
"""

device = torch.device("cuda")

class LSTMEncoder(nn.Module):
    def __init__(self, args):
        super(LSTMEncoder, self).__init__()
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
        super(RandomSystematicLinearEncoding, self).__init__()
        matrix = torch.randint(0, args.alphabet_size, (args.message_length, args.code_length - args.message_length)).float()
        self.register_buffer('matrix', matrix)
        self.alphabet_size = args.alphabet_size

    def forward(self, x):
        parity_bits = torch.remainder(x.float() @  self.matrix, self.alphabet_size)
        return torch.cat([x, parity_bits], dim=1).long()


class RandomLinearEncoding(nn.Module):
    def __init__(self, args):
        super(RandomLinearEncoding, self).__init__()
        matrix = torch.randint(0, args.alphabet_size, (args.message_length, args.code_length)).float()
        self.register_buffer('matrix', matrix)
        self.alphabet_size = args.alphabet_size

    def forward(self, x):
        return (x.float() @ self.matrix).long()

