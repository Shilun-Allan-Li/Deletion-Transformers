# -*- coding: utf-8 -*-
from datetime import datetime
import logging
import os
import sys
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


torch.backends.cudnn.benchmark = True

device = torch.device("cuda")

parser = argparse.ArgumentParser(description='Deep Deletion Code')

# General
parser.add_argument('--alphabet_size', type=int, default=2,
                    help='Size of the code alphabet (default: 2)')
parser.add_argument('--code_length', type=int, default=128,
                    help='Length of deletion code (default: 128)')
parser.add_argument('--message_length', type=int, default=64,
                    help='Length message (default: 64)')
parser.add_argument('--channel_prob', type=float, default=0.1,
                    help='Probability of channel (default: 0.1)')
parser.add_argument('--log_name', type=str, default="DDC_train",
                    help='Name of the log file (default: DDC_train)')

# parser.add_argument('--transition_prob', type=float, default=0.11, metavar='N',
#                     help='Cross transition probability of markov code (default: 0.11)')


# Training args
parser.add_argument('--batch_size', type=int, default=256,
                    help='input batch size for training (default: 64)')
parser.add_argument('--steps', type=int, default=400000,
                    help='number of epochs to train (default: 40000)')
parser.add_argument('--gamma', type=float, default=0.7,
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--num_sample', type=int, default=16,
                    help='Number of samples for encoder output distribution of codewords (default: 16)')
parser.add_argument('--clip', type=float, default=1,
                    help='training grad norm clip value (default: 1.0)')


# Testing args
parser.add_argument('--test_size', type=int, default=1000,
                    help='total samples to test (default: 1000)')
parser.add_argument('--dry_run', action='store_true', default=False,
                    help='quickly check a single pass')

# Encoder args
parser.add_argument('--encoder_lr', type=float, default=0.001, metavar='ELR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--train_encoder', type=bool, default=False,
                    help='Whether the encoder requires training (default: False)')
# parser.add_argument('--encoder_hidden', type=int, default=128,
#                     help='hidden layer dimension of encoder (default: 128)')

# Decoder args
parser.add_argument('--decoder_lr', type=float, default=1e-3, metavar='DLR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--decoder_e_hidden', type=int, default=8,
                    help='decoder hidden size (default: 8)')
parser.add_argument('--decoder_d_hidden', type=int, default=16,
                    help='decoder hidden size (default: 16)')
parser.add_argument('--decoder_forward', type=int, default=32,
                    help='decoder feed forward size (default: 32)')



# Output
# parser.add_argument('--checkpoint_name', type=str, default='DDC code',
#                     help="name of saved checkpoint (default: DCC code)")
parser.add_argument('--log_interval', type=int, default=10,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_model', action='store_true', default=False,
                    help='For Saving the current Model')

# Misc
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()

log_dir = "../runs/{}".format(datetime.now().strftime("%m%d %H-%M-%S"))
os.makedirs(log_dir, exist_ok=True)

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
file_handler = logging.FileHandler('{}/{}.log'.format(log_dir, args.log_name), 'w')
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

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

    src_padding_mask = (src == pad_token).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_token).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def train(args, encoder, decoder, E_optimizer, D_optimizer):
    pad_token = args.alphabet_size
    bos_token = args.alphabet_size + 1
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token)

    if args.train_encoder:
        encoder.train()
    decoder.train()
    for step in range(1, args.steps+1):
        if args.train_encoder:
            E_optimizer.zero_grad()
        D_optimizer.zero_grad()
        
        message = torch.randint(0, args.alphabet_size, (args.batch_size, args.message_length), device=device)

        ### for traditional encoder with bool output
        codeword_samples = encoder(message)

        ### for normal encoder with float output
        # codeword_dist = encoder(message)
        # codeword_samples = [codeword_dist[i].multinomial(num_samples=args.num_sample, replacement=True).transpose(0, 1) for i in range(args.batch_size)]
        # codeword_samples = torch.cat(codeword_samples)

        # codeword_samples is of size (batchsize * num_sample, codeword length)
        x, src_len = BSCChannel(codeword_samples, args.channel_prob)
        
        src = x.transpose(0, 1)
        tgt = message.transpose(0, 1)
        
        # append bos token at the start
        src = torch.cat([torch.tensor([[bos_token]*args.batch_size], device=device), src], dim=0)
        tgt = torch.cat([torch.tensor([[bos_token]*args.batch_size], device=device), tgt], dim=0)
        
        tgt_input = tgt[:-1, :]
        
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, pad_token)
        
        output = decoder(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        
        tgt_out = tgt[1:, :]
        
        # output = decoder(x)

        if step % 100 == 0:
            pass
        predictions = output.argmax(-1)
        
        BLER = torch.mean(torch.all(predictions == tgt_out, dim=0).float())
        BER = torch.mean((predictions == tgt_out).float())
        
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        
        loss = criterion(output, tgt_out.contiguous().view(-1))
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.clip)

        if args.train_encoder:
            E_optimizer.step()
        D_optimizer.step()
        
        writer.add_scalar('train/Loss', loss.item(), step)
        writer.add_scalar('train/BLER', BLER, step)
        writer.add_scalar('train/BER', BER, step)
        
        logger.info("[train] Step: {}/{} ({:.0f}%)\tLoss: {:.6f}\t BER: {}\t BLER: {}"
                    .format(step, args.steps, step/args.steps*100, loss.item(), BER, BLER))


def greedy_decode(args, model, src):
    # src should be (src_length, batch_size)
    batch_size = src.size(1)
    pad_token = args.alphabet_size
    bos_token = args.alphabet_size + 1
    
    src_padding_mask = (src == pad_token).transpose(0, 1)
    
    outputs = torch.zeros(args.message_length, batch_size, model.output_dim, device=device)
    memory = model.encode(src, None, src_padding_mask)

    # first input to the decoder is the <sos> tokens
    ys = torch.tensor([[bos_token]*batch_size], device=device)

    # mask = [batch size, src len]

    for t in range(args.message_length):
        # insert input token embedding, previous hidden state, all encoder hidden states
        #  and mask
        # receive output tensor (predictions) and new hidden state
        
        # tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(device)
        tgt_mask = None
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
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token)
    
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        for step in range(5):
            message = torch.randint(0, args.alphabet_size, (args.batch_size, args.message_length), device=device)
    
            ### for traditional encoder with bool output
            codeword_samples = encoder(message)
    
            # codeword_samples is of size (batchsize * num_sample, codeword length)
            x, src_len = BSCChannel(codeword_samples, args.channel_prob)
            
            src = x.transpose(0, 1)
            tgt = message.transpose(0, 1)
            
            # append bos token at the start
            src = torch.cat([torch.tensor([[bos_token]*args.batch_size], device=device), src], dim=0)
            
            output = greedy_decode(args, decoder, src)
    
            if step % 100 == 0:
                pass
            predictions = output.argmax(-1)
            
            BLER = torch.mean(torch.all(predictions == tgt, dim=0).float())
            BER = torch.mean((predictions == tgt).float())
            
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            
            loss = criterion(output, tgt.contiguous().view(-1))
            
            writer.add_scalar('test/Loss', loss.item(), step)
            writer.add_scalar('test/BLER', BLER, step)
            writer.add_scalar('test/BER', BER, step)
            
            logger.info("[test] Step: {}/{} ({:.0f}%)\tLoss: {:.6f}\t BER: {}\t BLER: {}"
                        .format(step, args.steps, step/args.steps*100, loss.item(), BER, BLER))
            
    if args.train_encoder:
            encoder.train()
    decoder.train()


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
    
    # train_dataset, train_loader = data_loader(
    #     dataset_size=args.steps*args.batch_size,
    #     length=args.length,
    #     deletion_p=args.channel_prob,
    #     transition_p=args.transition_prob,
    #     batch_size=args.batch_size)
    
    # test_dataset, test_loader = data_loader(
    #     dataset_size=args.test_size,
    #     length=args.length,
    #     deletion_p=args.channel_prob,
    #     transition_p=args.transition_prob,
    #     batch_size=args.test_size)

    def init_weights(m):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)

    encoder = RandomSystematicLinearEncoding(args).to(device)
    decoder = Seq2SeqTransformer(args).to(device)
    # decoder = testDecoder(args).to(device)
    # decoder.apply(init_weights)
    for p in decoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
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