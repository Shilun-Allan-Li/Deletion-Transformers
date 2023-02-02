# -*- coding: utf-8 -*-
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from models import *
import random


"""
The input to the decoder is tensor of shape (batch, code_length) and Long type on cuda
The output of the decoder should be a tensor of shape (batch, message_length)
"""

device = torch.device("cuda")


class Seq2SeqTransformer(nn.Module):
    def __init__(self, args):
        super(Seq2SeqTransformer, self).__init__()
        src_vocab_size = args.alphabet_size+2
        tgt_vocab_size = args.alphabet_size+2
        self.vocab_size = args.alphabet_size+2 # with <pad> and <bos>
        self.pad_token = args.alphabet_size
        self.bos_token = args.alphabet_size+1
        self.output_dim = tgt_vocab_size
        emb_size = 8
        nhead = emb_size
        dropout=0
        
        self.transformer = nn.Transformer(d_model=emb_size,
                                          nhead=nhead,
                                          num_encoder_layers=4,
                                          num_decoder_layers=4,
                                          dim_feedforward=32,
                                          dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

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
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor, src_padding_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask, src_padding_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)


class Seq2SeqGRU(nn.Module):
    def __init__(self, args):
        super(Seq2SeqGRU, self).__init__()
        self.vocab_size = args.alphabet_size+2 # with <pad> and <bos>
        self.pad_token = args.alphabet_size
        self.bos_token = args.alphabet_size+1
        self.message_length = args.message_length
        self.encoder = SeqEncoder(input_dim=self.vocab_size,
                                  emb_dim=self.vocab_size,
                                  enc_hid_dim=args.decoder_e_hidden,
                                  dec_hid_dim=args.decoder_d_hidden)

        self.decoder = SeqDecoder(input_dim=self.vocab_size,
                                  output_dim=args.alphabet_size,
                                  emb_dim=self.vocab_size,
                                  enc_hid_dim=args.decoder_e_hidden,
                                  dec_hid_dim=args.decoder_d_hidden)
    def create_mask(self, src):
        mask = (src != self.pad_token).permute(1, 0)
        return mask
    
    
    def forward(self, src, src_len, trg, teacher_forcing_ratio):
        # x is of shape (codeword length, N, dictionary_size)
        batch_size = src.shape[1]
        outputs = torch.zeros(self.message_length, batch_size, self.decoder.output_dim, device=device)
        encoder_outputs, hidden = self.encoder(src, src_len)

        # first input to the decoder is the <sos> tokens
        input = torch.tensor([self.bos_token]*batch_size, device=device)

        mask = self.create_mask(src)

        # mask = [batch size, src len]

        for t in range(self.message_length):
            # insert input token embedding, previous hidden state, all encoder hidden states
            #  and mask
            # receive output tensor (predictions) and new hidden state

            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs


class SimpleDecoder(nn.Module):
    def __init__(self, args):
        super(SimpleDecoder, self).__init__()
        self.encoder = nn.GRU(args.alphabet_size+1, args.decoder_hidden)
        self.decoder = nn.GRU(args.alphabet_size+1, args.decoder_hidden)
        self.out = nn.Linear(args.decoder_hidden, args.alphabet_size+1)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, x, src_len):
        # x is batch first with shape (padded sequence length, batch size, alphabet size+1)
        # idxs = torch.argsort(src_len, descending=True)
        # x = x[:,idxs]
        # src_len = src_len[idxs]
        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_len.to('cpu'), enforce_sorted=False)
        packed_x_features, encoder_hidden = self.encoder(packed_x)
        encoder_output, _ = nn.utils.rnn.pad_packed_sequence(packed_x_features, total_length=args.code_length)
        output, hidden = self.decoder(x, encoder_hidden)
        pass
        output = self.softmax(self.out(output))
        return
    
class testDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.emb = nn.Embedding(2, 2)
        # self.conv = nn.Conv1d(2, 2, kernel_size=1, padding=0)
        self.fc = nn.Linear(128, 64)
        
    def forward(self, x):
        embedding = self.emb(x)
        out = self.fc(embedding.transpose(1, 2)).transpose(1, 2)
        return out.transpose(0, 1)
        # out = self.conv(embedding.transpose(1, 2))
        # return out.transpose(1, 2)