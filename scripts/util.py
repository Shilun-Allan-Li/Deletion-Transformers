# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

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
        substitutionCost = v0[:, :-1] + (x != y[:, i, None]).int()
        v1[:, 1:] = torch.minimum(v1[:, 1:], substitutionCost)
        v1[:, 1:] = torch.minimum(v1[:, 1:], deletionCost)
        v0 = v1

    return v0[:, -1]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def generate_square_subsequent_mask(sz, independent):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return (mask + mask.transpose(0, 1)) if independent else mask

def create_mask(src=None, tgt=None, pad_token=None, independent=True):
    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = None, None, None, None
    if src is not None:
        src_seq_len = src.shape[0]
        src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)
        src_padding_mask = torch.all(src == pad_token, dim=2).transpose(0, 1)
    
    if tgt is not None:
        tgt_seq_len = tgt.shape[0]
        tgt_mask = generate_square_subsequent_mask(tgt_seq_len, independent)
    return src_mask, tgt_mask, src_padding_mask, None

def mod_relu(x):
    return torch.remainder(F.relu(x), 2) - 1

def sin_activation(x):
    return torch.sin(x/10)

def greedy_decode(message_length, model, src, bos_token, pad_token, independent):
    # src should be (src_length, batch_size)
    batch_size = src.size(1)
    
    _, _, src_padding_mask, _ = create_mask(src=src, pad_token=pad_token)
    
    outputs = torch.zeros(message_length, batch_size, model.output_dim, device=device)
    memory = model.encode(src, None, src_padding_mask)

    # first input to the decoder is the <sos> tokens
    ys = torch.tensor([[bos_token]*batch_size], device=device)

    # mask = [batch size, src len]

    for t in range(message_length):
        # insert input token embedding, previous hidden state, all encoder hidden states
        #  and mask
        # receive output tensor (predictions) and new hidden state
        
        # use when target bits are independent
        _, tgt_mask, _, _ = create_mask(tgt=ys, independent=independent)
        
        # tgt_mask = None
        features = model.decode(ys, memory, tgt_mask)
        features = features.transpose(0, 1)
        output = model.generator(features[:, -1])

        # place predictions in a tensor holding predictions for each token
        outputs[t] = output

        # get the highest predicted token from our predictions
        top1 = output.argmax(1)
        ys = torch.cat([ys, torch.unsqueeze(top1, 0)], dim=0)
    return outputs