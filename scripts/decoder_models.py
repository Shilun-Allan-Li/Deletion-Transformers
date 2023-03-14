# -*- coding: utf-8 -*-
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from custom_transformers import Transformer
# from torch.nn import Transformer
import random


"""
The input to the decoder is tensor of shape (batch, code_length) and float type on cuda
The output of the decoder should be a tensor of shape (batch, message_length, logits size)
"""

device = torch.device("cuda")

class Seq2SeqTransformer(nn.Module):
    def __init__(self, args, output_dim):
        super(Seq2SeqTransformer, self).__init__()
        self.output_dim = output_dim
        tgt_vocabsize = args.alphabet_size+1
        emb_size = 8
        nhead = emb_size
        dropout=0.1
        
        self.transformer =  Transformer(d_model=emb_size,
                                        nhead=nhead,
                                        activation='relu',
                                        num_encoder_layers=1,
                                        num_decoder_layers=1,
                                        dim_feedforward=8,
                                        dropout=dropout)
        self.generator = nn.Linear(emb_size, output_dim)
        self.src_tok_emb = nn.Linear(1, emb_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocabsize, emb_size)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        # src_emb = self.positional_encoding(self.src_tok_emb(src))
        # tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        src_emb = self.src_tok_emb(src)
        tgt_emb = self.tgt_tok_emb(trg)
        
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor, src_padding_mask: Tensor):
        # src_emb = self.positional_encoding(self.src_tok_emb(src))
        src_emb = self.src_tok_emb(src)
        return self.transformer.encoder(src_emb, src_mask, src_padding_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        # tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        tgt_emb = self.tgt_tok_emb(tgt)
        return self.transformer.decoder(tgt_emb, memory, tgt_mask)


class ConvDecoder(nn.Module):
    def __init__(self, args):
        super(ConvDecoder, self).__init__()
        self.model = nn.Sequential(
                      nn.Conv1d(2, 256, kernel_size=5, padding=2),
                      nn.ReLU(),
                      nn.Conv1d(256, 128, kernel_size=3, padding=1),
                      nn.ReLU(),
                      nn.Conv1d(128, 128, kernel_size=3, padding=1),
                      nn.ReLU(),
                      nn.Conv1d(128, 128, kernel_size=3, padding=1),
                      nn.ReLU(),
                      nn.Conv1d(128, 64, kernel_size=3, padding=1),
                      nn.ReLU(),
                      nn.Conv1d(64, 64, kernel_size=3, padding=1),
                      nn.ReLU(),
                      nn.Conv1d(64, 64, kernel_size=3, padding=1),
                      nn.ReLU(),
                      nn.Conv1d(64, 32, kernel_size=3, padding=1),
                      nn.ReLU(),
                      nn.Conv1d(32, 32, kernel_size=3, padding=1),
                      nn.ReLU(),
                      nn.Conv1d(32, 32, kernel_size=3, padding=1),
                      nn.ReLU(),
                      nn.Conv1d(32, 2, kernel_size=3, padding=1)
                    )
        
    def forward(self, x):
        x = x.float().reshape(x.size(0), 2, x.size(1)//2)
        return self.model(x).transpose(1, 2)