# -*- coding: utf-8 -*-
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from models import *
import random


class Seq2SeqDecoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.vocab_size = args.alphabet_size+2 # with <pad> and <bos>
        self.bos_token = args.alphabet_size+1
        self.message_length = args.message_length
        self.encoder = SeqEncoder(input_dim=self.vocab_size,
                                  emb_dim=self.vocab_size,
                                  enc_hid_dim=args.decoder_e_hidden,
                                  dec_hid_dim=args.decoder_d_hidden)

        self.decoder = SeqDecoder(output_dim=self.vocab_size,
                                  emb_dim=self.vocab_size,
                                  enc_hid_dim=args.decoder_e_hidden,
                                  dec_hid_dim=args.decoder_d_hidden)

    def forward(self, src, src_len, trg, teacher_forcing_ratio):
        # x is of shape (codeword length, N, dictionary_size)
        batch_size = src.shape[1]
        outputs = torch.zeros(self.message_length, batch_size, self.vocab_size, device=device)
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
        super(Decoder, self).__init__()
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