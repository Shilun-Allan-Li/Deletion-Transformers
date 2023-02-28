# -*- coding: utf-8 -*-
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import random

device = torch.device("cuda")

class ConvAE(nn.Module):
    def __init__(self, args):
        super(ConvAE, self).__init__()
        pass

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        # src_emb = self.src_tok_emb(src)
        # tgt_emb = self.tgt_tok_emb(trg)
        
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor, src_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        # src_emb = self.src_tok_emb(src)
        return self.transformer.encoder(src_emb, src_mask, src_padding_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        # tgt_emb = self.tgt_tok_emb(tgt)
        return self.transformer.decoder(tgt_emb, memory, tgt_mask)