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

class ConvEncoder(nn.Module):
    def __init__(self, args):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv1d(1, 256, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(64, 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.layernorm = nn.LayerNorm(args.code_length, elementwise_affine=False)
        

    def forward(self, x):
        # x has size (batch, message length)
        N, message_length = x.size(0), x.size(1)
        x = torch.unsqueeze(x.float(), 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)    
        out = x.reshape(N, message_length*2)
        return self.layernorm(out)

class LSTMEncoder(nn.Module):
    def __init__(self, args):
        super(LSTMEncoder, self).__init__()
        self.emb = nn.Embedding(args.alphabet_size, args.alphabet_size)
        self.linear = nn.Conv1d(args.message_length, args.code_length, kernel_size=1, padding=0)
        # self.lstm = nn.LSTM(args.alphabet_size, args.alphabet_size, batch_first=True)
        self.lstm = nn.LSTM(args.alphabet_size, 1, batch_first=True)
        self.softmax = nn.Softmax(dim=-1)
        

    def forward(self, x):
        # x has size (batch, message length)
        x = self.emb(x)
        hidden = self.linear(x)
        logits, _ = self.lstm(hidden)
        # output size is (batch, code length, alphabet size)
        # output = self.softmax(logits)
        output = logits
        return output
    
class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        self.emb = nn.Embedding(args.alphabet_size, args.alphabet_size)
        self.linear = nn.Conv1d(args.message_length, args.code_length, kernel_size=1, padding=0)
        # self.lstm = nn.LSTM(args.alphabet_size, args.alphabet_size, batch_first=True)
        self.lstm = nn.LSTM(args.alphabet_size, 1, batch_first=True)
        self.softmax = nn.Softmax(dim=-1)
        

    def forward(self, x):
        # x has size (batch, message length)
        x = self.emb(x)
        hidden = self.linear(x)
        logits, _ = self.lstm(hidden)
        # output size is (batch, code length, alphabet size)
        # output = self.softmax(logits)
        output = logits
        return output


class RandomSystematicLinearEncoding(nn.Module):
    def __init__(self, args):
        super(RandomSystematicLinearEncoding, self).__init__()
        # matrix = torch.randint(0, args.alphabet_size, (args.message_length, args.code_length - args.message_length)).float()
        # matrix = torch.zeros((args.message_length, args.code_length - args.message_length))
        sample = torch.rand((args.message_length, args.code_length - args.message_length)).topk(3, dim=1).indices
        mask = torch.zeros((args.message_length, args.code_length - args.message_length))
        matrix = mask.scatter_(dim=1, index=sample, value=1)
        matrix = matrix.float()
        self.register_buffer('matrix', matrix)
        self.alphabet_size = args.alphabet_size

    def forward(self, x):
        parity_bits = torch.remainder(x.float() @  self.matrix, self.alphabet_size)
        return torch.cat([x, parity_bits], dim=1).long()


class RandomLinearEncoding(nn.Module):
    def __init__(self, args):
        super(RandomLinearEncoding, self).__init__()
        # matrix = torch.randint(0, args.alphabet_size, (args.message_length, args.code_length)).float()
        # matrix = (torch.rand((args.message_length, args.code_length)) < 0.02).float()
        sample = torch.rand((args.message_length, args.code_length-18)).topk(3, dim=1).indices
        mask = torch.zeros((args.message_length, args.code_length-18))
        matrix = mask.scatter_(dim=1, index=sample, value=1)
        m = torch.diag(torch.ones(args.message_length-1), diagonal=1) + torch.diag(torch.ones(args.message_length)) + torch.diag(torch.ones(args.message_length-1), diagonal=-1)
        m = torch.cat([torch.tensor([[1] + [0]*15]), m, torch.tensor([[0]*15 + [1]])], dim=0).transpose(0, 1)
        matrix = torch.cat([m, matrix], dim=1)
        matrix = matrix.float()
        self.register_buffer('matrix', matrix)
        self.alphabet_size = args.alphabet_size

    def forward(self, x):
        return torch.remainder(x.float() @ self.matrix, self.alphabet_size).long()
    
class RepetitionCode(nn.Module):
    def __init__(self, args):
        super(RepetitionCode, self).__init__()
        assert args.code_length % args.message_length == 0
        self.repeat = args.code_length // args.message_length

    def forward(self, x):
        return torch.cat([x]*self.repeat, dim=1)
    
class PolarCode(nn.Module):
    def __init__(self, args, systematic):
        super(PolarCode, self).__init__()
        if systematic:
            matrix = torch.load("weights/polar_systematic_{}x{}.pt".format(args.code_length, args.message_length)).transpose(0, 1).float()
        else:
            matrix = torch.load("weights/polar_{}x{}.pt".format(args.code_length, args.message_length)).transpose(0, 1).float()
        self.register_buffer('matrix', matrix)
        
    def forward(self, x):
        return torch.remainder(x.float() @ self.matrix, 2).long()

