# -*- coding: utf-8 -*-
import math
import torch
from torch import nn
import torch.nn.functional as F
import encoder_models
import decoder_models
from util import create_mask, greedy_decode

device = torch.device("cuda")

class ConvAE(nn.Module):
    def __init__(self, args, channel):
        super(ConvAE, self).__init__()
        self.encoder = encoder_models.ConvEncoder(args)
        self.decoder = decoder_models.ConvDecoder(args)
        self.channel = channel


    def forward(self, message):
        codeword = self.encoder(torch.pow(-1, message))
        x, src_len = self.channel(codeword) 
        output = self.decoder(x)
        return output
    
    def predict(self, message):
        return self.forward(message)
    


class ConvTransformerAE(nn.Module):
    def __init__(self, args, channel):
        super(ConvTransformerAE, self).__init__()
        self.encoder = encoder_models.ConvEncoder(args)
        self.decoder = decoder_models.Seq2SeqTransformer(args, output_dim=2)
        self.channel = channel
        
        self.pad_token = 0
        self.bos_token = args.alphabet_size
        self.message_length = args.message_length

    def forward(self, message):
        codeword = self.encoder(torch.pow(-1, message))
        x, src_len = self.channel(codeword)
        
        src = x.transpose(0, 1).unsqueeze(2)
        tgt = message.transpose(0, 1)
        tgt = torch.cat([torch.tensor([[self.bos_token]*codeword.size(0)], device=device), tgt], dim=0)
        
        tgt_input = tgt[:-1, :]
        
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, self.pad_token)
        
        output = self.decoder(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        
        return output.transpose(0, 1)
    
    def predict(self, message):
        codeword = self.encoder(torch.pow(-1, message))
        x, src_len = self.channel(codeword)
        
        src = x.transpose(0, 1).unsqueeze(2)
        
        output = greedy_decode(self.message_length, self.decoder, src, self.bos_token, pad_token=0, independent=True)
        return output.transpose(0, 1)