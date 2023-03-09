# -*- coding: utf-8 -*-
from datetime import datetime
import logging
import os
import sys
import numpy as np
import torch
import argparse
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from channels import *
from encoder_models import *
from decoder_models import *
import shutil


torch.backends.cudnn.benchmark = True

device = torch.device("cuda")

parser = argparse.ArgumentParser(description='Deep Deletion Code')

checkpoint_path = None
# checkpoint_path = '../runs/32 repete 4 times/checkpoint.pt'

# General
parser.add_argument('--log_name', type=str, default="AWGN 100 to 200 SNR 0 Conv encoder",
                    help='Name of the log folder (default: current time)')
parser.add_argument('--checkpoint_load_path', type=str, default=checkpoint_path,
                    help='checkpoint path to load (default: None)')
parser.add_argument('--alphabet_size', type=int, default=2,
                    help='Size of the code alphabet (default: 2)')
parser.add_argument('--code_length', type=int, default=200,
                    help='Length of deletion code (default: 128)')
parser.add_argument('--message_length', type=int, default=100,
                    help='Length message (default: 64)')
parser.add_argument('--channel_prob', type=float, default=0.1,
                    help='Probability of channel (default: 0.1)')
parser.add_argument('--SNR', type=float, default=0,
                    help='SNR of AWGN channel (default: 0.0)')


# Training args
parser.add_argument('--batch_size', type=int, default=256,
                    help='input batch size for training (default: 256)')
parser.add_argument('--steps', type=int, default=100000,
                    help='number of epochs to train (default: 100000)')
parser.add_argument('--gamma', type=float, default=0.7,
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--num_sample', type=int, default=16,
                    help='Number of samples for encoder output distribution of codewords (default: 16)')
parser.add_argument('--clip', type=float, default=1,
                    help='training grad norm clip value (default: 1.0)')

# Testing args
parser.add_argument('--eval_size', type=int, default=2048,
                    help='total samples to test (default: 1024)')
parser.add_argument('--eval_every', type=int, default=500,
                    help='eval every n steps (default: 1000)')

# Encoder args
parser.add_argument('--encoder_lr', type=float, default=1e-3, metavar='ELR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--train_encoder', type=bool, default=True,
                    help='Whether the encoder requires training (default: False)')
# parser.add_argument('--encoder_hidden', type=int, default=128,
#                     help='hidden layer dimension of encoder (default: 128)')

# Decoder args
parser.add_argument('--decoder_lr', type=float, default=1e-3, metavar='DLR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--decoder_e_hidden', type=int, default=8,
                    help='decoder hidden size (default: 8)')
parser.add_argument('--decoder_d_hidden', type=int, default=16,
                    help='decoder hidden size (default: 16)')
parser.add_argument('--decoder_forward', type=int, default=32,
                    help='decoder feed forward size (default: 32)')

# Output
parser.add_argument('--save_model', action='store_true', default=True,
                    help='For Saving the current Model')

# Misc
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()

log_dir = "../runs/{}".format(args.log_name if args.log_name else datetime.now().strftime("%m%d %H-%M-%S"))
if os.path.exists(os.path.join(log_dir, "tensorboard")):
    s = str(input("log directory already exists. Continue? [Y/N]"))
    s = s.lower()
    if s != "y" and s != "yes":
        os._exit(0)
    shutil.rmtree(os.path.join(log_dir, "tensorboard"))
    
os.makedirs(log_dir, exist_ok=True)

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
file_handler = logging.FileHandler('{}/log.txt'.format(log_dir), 'w')
stdout_handler = logging.StreamHandler(stream=sys.stdout)
logging.basicConfig(level=logging.INFO, format=FORMAT, handlers=[file_handler, stdout_handler], force=True)
logger = logging.getLogger(__name__)



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, pad_token):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]
    
    # use when target bits are independent
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len) + generate_square_subsequent_mask(tgt_seq_len).transpose(0, 1)
    
    # use when target bits depend on the former bits
    # tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)
    pad_tensor = F.one_hot(torch.tensor([pad_token], device=device), src.size(2)).float()
    src_padding_mask = torch.all(src == pad_tensor, dim=2).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_token).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def train(args, encoder, decoder, E_optimizer, D_optimizer):
    pad_token = args.alphabet_size
    bos_token = args.alphabet_size + 1
    vocab_size = args.alphabet_size + 2
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token)
    best_BER = 1

    if args.train_encoder:
        encoder.train()
    decoder.train()
    for step in range(1, args.steps+1):
        if args.train_encoder:
            E_optimizer.zero_grad()
        D_optimizer.zero_grad()
        
        message = torch.randint(0, args.alphabet_size, (args.batch_size, args.message_length), device=device)

        # if args.train_encoder:
        #     ### for trainable encoder with float output
        #     codeword_dist = encoder(message)
        #     codeword_samples_idx = [codeword_dist[i].multinomial(num_samples=args.num_sample, replacement=True).transpose(0, 1) for i in range(args.batch_size)]
        #     codeword_samples_idx = torch.cat(codeword_samples_idx)
        #     codeword_samples = F.one_hot(codeword_samples_idx, vocab_size).float()
        #     codeword_samples.requires_grad = True
        #     message = torch.repeat_interleave(message, args.num_sample, 0)
        # else:
        #     ### for traditional encoder with bool output
        #     codeword_samples_idx = encoder(message)
        #     codeword_samples = F.one_hot(codeword_samples_idx, vocab_size).float()
        
        """test AWGN"""
        codeword = encoder(torch.pow(-1, message))
        codeword_samples = torch.zeros((args.batch_size, args.code_length, 4), device=device)
        codeword_samples[:, :, 0] = codeword
        
        x, src_len = AWGN(codeword_samples, args.SNR)
        
        # codeword_samples is of size (batchsize * num_sample, codeword length)
        # x, src_len = BSCChannel(codeword_samples, args.channel_prob)
        # x, src_len = deletionChannel(codeword_samples, args.channel_prob, pad_token, vocab_size)
        
        src = x.transpose(0, 1)
        tgt = message.transpose(0, 1)
        
        # append bos token at the start
        src = torch.cat([F.one_hot(torch.tensor([[bos_token]*codeword_samples.size(0)], device=device), vocab_size).float(), src], dim=0)
        tgt = torch.cat([torch.tensor([[bos_token]*codeword_samples.size(0)], device=device), tgt], dim=0)
        
        tgt_input = tgt[:-1, :]
        
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, pad_token)
        
        output = decoder(src.float(), tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        
        # output = decoder(x)
        
        predictions = output.argmax(-1)
        
        BLER = torch.mean(torch.any(predictions != tgt_out, dim=0).float())
        BER = torch.mean((predictions != tgt_out).float())
        
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        
        loss = criterion(output, tgt_out.contiguous().view(-1))
        loss.backward()

        # if args.train_encoder:
        #     grad = torch.mean(codeword_samples.grad.view(args.batch_size, args.num_sample, args.code_length, vocab_size), dim=1)[:,:,:args.alphabet_size]
        #     torch.autograd.backward(codeword_dist, grad)
        #     E_optimizer.step()
        E_optimizer.step()
        D_optimizer.step()
        
        writer.add_scalar('train/Loss', loss.item(), step)
        writer.add_scalar('train/BLER', BLER, step)
        writer.add_scalar('train/BER', BER, step)
        
        logger.info("[train] Step: {}/{} ({:.0f}%)\tLoss: {:.6f}\t BER: {}\t BLER: {}"
                    .format(step, args.steps, step/args.steps*100, loss.item(), BER, BLER))
        
        if step % args.eval_every == 0:
            logger.info("evaluating...")
            e_loss, e_BER, e_BLER = test(args, encoder, decoder)
            writer.add_scalar('eval/Loss', e_loss, step)
            writer.add_scalar('eval/BLER', e_BLER, step)
            writer.add_scalar('eval/BER', e_BER, step)
            
            logger.info("[eval] Step: {}/{} ({:.0f}%)\tLoss: {:.6f}\t BER: {}\t BLER: {}"
                        .format(step, args.steps, step/args.steps*100, e_loss, e_BER, e_BLER))
            
            if e_BER < best_BER and args.save_model:
                logger.info("saving model...")
                best_BER = e_BER
                checkpoint = {
                    'args': args.__dict__,
                    'encoder_name': encoder.__class__.__name__,
                    'encoder_state': encoder.state_dict(),
                    'decoder_name': decoder.__class__.__name__,
                    'decoder_state': decoder.state_dict(),
                    'step': step,
                    'loss': e_loss,
                    'BER': e_BER,
                    'BLER': e_BLER,
                    }
                checkpoint_file = os.path.join(log_dir, "checkpoint.pt")
                torch.save(checkpoint, checkpoint_file)
                logger.info("model saved to {}".format(checkpoint_file))
                
                

def greedy_decode(args, model, src):
    # src should be (src_length, batch_size)
    batch_size = src.size(1)
    pad_token = args.alphabet_size
    bos_token = args.alphabet_size + 1
    pad_tensor = F.one_hot(torch.tensor([pad_token], device=device), src.size(2)).float()
    
    src_padding_mask = torch.all(src == pad_tensor, dim=2).transpose(0, 1)
    
    outputs = torch.zeros(args.message_length, batch_size, model.output_dim, device=device)
    memory = model.encode(src, None, src_padding_mask)

    # first input to the decoder is the <sos> tokens
    ys = torch.tensor([[bos_token]*batch_size], device=device)

    # mask = [batch size, src len]

    for t in range(args.message_length):
        # insert input token embedding, previous hidden state, all encoder hidden states
        #  and mask
        # receive output tensor (predictions) and new hidden state
        
        # use when target bits are independent
        tgt_mask = generate_square_subsequent_mask(ys.size(0)) + generate_square_subsequent_mask(ys.size(0)).transpose(0, 1)
        
        # use when target bits depend on the former bits
        # tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(device)
        
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



def test(args, encoder, decoder):
    pad_token = args.alphabet_size
    bos_token = args.alphabet_size + 1
    vocab_size = args.alphabet_size + 2
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token)
    
    encoder.eval()
    decoder.eval()
    
    losses, BERs, BLERs = [], [], []
    
    with torch.no_grad():
        for step in range(args.eval_size // args.batch_size):
            message = torch.randint(0, args.alphabet_size, (args.batch_size, args.message_length), device=device)
            
            # if args.train_encoder:
            #     ### for trainable encoder with float output
            #     codeword_dist = encoder(message)
            #     codeword_samples_idx = [codeword_dist[i].multinomial(num_samples=args.num_sample, replacement=True).transpose(0, 1) for i in range(args.batch_size)]
            #     codeword_samples_idx = torch.cat(codeword_samples_idx)
            #     codeword_samples = F.one_hot(codeword_samples_idx, vocab_size).float()
            #     codeword_samples.requires_grad = True
            #     message = torch.repeat_interleave(message, args.num_sample, 0)
            # else:
            #     ### for traditional encoder with bool output
            #     codeword_samples_idx = encoder(message)
            #     codeword_samples = F.one_hot(codeword_samples_idx, vocab_size).float()

            # # codeword_samples is of size (batchsize * num_sample, codeword length)
            # x, src_len = deletionChannel(codeword_samples, args.channel_prob, pad_token, vocab_size)
            
            
            
            """test AWGN"""
            codeword = encoder(torch.pow(-1, message))
            codeword_samples = torch.zeros((args.batch_size, args.code_length, 4), device=device)
            codeword_samples[:, :, 0] = codeword
            
            x, src_len = AWGN(codeword_samples, args.SNR)
            
            
            src = x.transpose(0, 1)
            tgt = message.transpose(0, 1)
            
            # append bos token at the start
            src = torch.cat([F.one_hot(torch.tensor([[bos_token]*codeword_samples.size(0)], device=device), vocab_size).float(), src], dim=0)
            
            output = greedy_decode(args, decoder, src.float())
    
            predictions = output.argmax(-1)
            
            BLER = torch.mean(torch.any(predictions != tgt, dim=0).float())
            BER = torch.mean((predictions != tgt).float())
            
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            
            loss = criterion(output, tgt.contiguous().view(-1))
            
            losses.append(loss.item())
            BLERs.append(BLER.cpu())
            BERs.append(BER.cpu())
            
    if args.train_encoder:
            encoder.train()
    decoder.train()

    return np.mean(losses), np.mean(BERs), np.mean(BLERs)
            
    

def main(args):
    # Training settings
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    logger.info('training with the following args:')
    logger.info("=" * 50)
    for k, v in sorted(list(args.__dict__.items())):
        logger.info("{}: {}".format(k, v))
    logger.info("=" * 50)
    
    logger.info('Training on {} datapoints with {} steps and batchsize {}'.format(args.steps*args.batch_size, args.steps, args.batch_size))

    encoder = ConvEncoder(args).to(device)
    # encoder = RepetitionCode(args).to(device)
    # encoder = PolarCode(args, True).to(device)
    # encoder = RandomSystematicLinearEncoding(args).to(device)
    # encoder = RandomLinearEncoding(args).to(device)
    decoder = Seq2SeqTransformer(args).to(device)
    # decoder = testDecoder2(args).to(device)
    
    # for p in decoder.parameters():
    #     if p.dim() > 1:
    #         nn.init.xavier_uniform_(p)
            
    if args.checkpoint_load_path is not None:
        logger.info("loading checkpoint from {}".format(args.checkpoint_load_path))
        checkpoint = torch.load(args.checkpoint_load_path)
        assert encoder.__class__.__name__ == checkpoint['encoder_name']
        assert decoder.__class__.__name__ == checkpoint['decoder_name']
        encoder.load_state_dict(checkpoint['encoder_state'])
        decoder.load_state_dict(checkpoint['decoder_state'])
        logger.info("checkpoint loaded step: {}, Loss: {}, BER: {}, BLER: {}"
                    .format(checkpoint['step'], checkpoint['loss'], checkpoint['BER'], checkpoint['BLER']))
            
    logger.info("The encoder has {} trainable parameters.".format(count_parameters(encoder)))
    logger.info("The decoder has {} trainable parameters.".format(count_parameters(decoder)))
    if args.train_encoder:
        E_optimizer = optim.Adam(encoder.parameters(), lr=args.encoder_lr)
    else:
        E_optimizer = None
    D_optimizer = optim.Adam(decoder.parameters(), lr=args.decoder_lr)
    train(args, encoder, decoder, E_optimizer, D_optimizer)
    test(args, encoder, decoder)


if __name__ == '__main__':
    writer = SummaryWriter(os.path.join(
        log_dir,
        "tensorboard"
    ))
    main(args)