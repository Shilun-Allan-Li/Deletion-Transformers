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
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import channels
import autoencoders
from util import editDistance, count_parameters
import shutil


torch.backends.cudnn.benchmark = True

device = torch.device("cuda")

parser = argparse.ArgumentParser(description='Deep Deletion Code')

checkpoint_path = None
# checkpoint_path = '../runs/32 repete 4 times/checkpoint.pt'

# General
parser.add_argument('--log_name', type=str, default="test2",
                    help='Name of the log folder (default: current time)')
parser.add_argument('--checkpoint_load_path', type=str, default=checkpoint_path,
                    help='checkpoint path to load (default: None)')
parser.add_argument('--alphabet_size', type=int, default=2,
                    help='Size of the code alphabet (default: 2)')
parser.add_argument('--code_length', type=int, default=200,
                    help='Length of deletion code (default: 128)')
parser.add_argument('--message_length', type=int, default=100,
                    help='Length message (default: 64)')
parser.add_argument('--channel_prob', type=float, default=0,
                    help='Probability of channel (default: 0.05)')
parser.add_argument('--SNR', type=float, default=4,
                    help='SNR of AWGN channel (default: 0.0)')


# Training args
parser.add_argument('--batch_size', type=int, default=1000,
                    help='input batch size for training (default: 256)')
parser.add_argument('--steps', type=int, default=4000,
                    help='number of epochs to train (default: 100000)')
parser.add_argument('--clip', type=float, default=1,
                    help='training grad norm clip value (default: 1.0)')

# Testing args
parser.add_argument('--eval_size', type=int, default=4000,
                    help='total samples to test (default: 1024)')
parser.add_argument('--eval_every', type=int, default=400,
                    help='eval every n steps (default: 1000)')

# Encoder args
parser.add_argument('--encoder_lr', type=float, default=1e-4, metavar='ELR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--train_encoder', type=bool, default=True,
                    help='Whether the encoder requires training (default: False)')
# parser.add_argument('--encoder_hidden', type=int, default=128,
#                     help='hidden layer dimension of encoder (default: 128)')

# Decoder args
parser.add_argument('--decoder_lr', type=float, default=1e-4, metavar='DLR',
                    help='learning rate (default: 1e-3)')
# parser.add_argument('--decoder_e_hidden', type=int, default=8,
#                     help='decoder hidden size (default: 8)')
# parser.add_argument('--decoder_d_hidden', type=int, default=16,
#                     help='decoder hidden size (default: 16)')
# parser.add_argument('--decoder_forward', type=int, default=32,
#                     help='decoder feed forward size (default: 32)')

# Output
parser.add_argument('--save_model', action='store_true', default=True,
                    help='For Saving the current Model')

# Misc
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')

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


def train(args, AE_model, E_optimizer, D_optimizer):
    criterion = nn.CrossEntropyLoss()
    best_BER = 1

    AE_model.train()
    for step in range(1, args.steps+1):
        if args.train_encoder:
            E_optimizer.zero_grad()
        D_optimizer.zero_grad()
        
        message = torch.randint(0, args.alphabet_size, (args.batch_size, args.message_length), device=device)
        
        output = AE_model(message).contiguous()
 
        predictions = output.argmax(-1)
        
        RLD = torch.mean(editDistance(predictions, message, None).float()) / message.size(1)
        BLER = torch.mean(torch.any(predictions != message, dim=1).float())
        BER = torch.mean((predictions != message).float())
        
        output_dim = output.shape[-1]
        
        loss = criterion(output.contiguous().view(-1, output_dim), message.view(-1))
        loss.backward()

        if args.train_encoder:
            E_optimizer.step()
        D_optimizer.step()
        
        writer.add_scalar('train/Loss', loss.item(), step)
        writer.add_scalar('train/BLER', BLER, step)
        writer.add_scalar('train/BER', BER, step)
        writer.add_scalar('train/RLD', RLD, step)
        
        logger.info("[train] Step: {}/{} ({:.0f}%)\tLoss: {:.6f}\t BER: {:.6f}\t RLD: {:.6f}\t BLER: {:.6f}"
                    .format(step, args.steps, step/args.steps*100, loss.item(), BER, RLD, BLER))
        
        if step % args.eval_every == 0:
            logger.info("evaluating...")
            e_loss, e_BER, e_BLER, e_RLD = test(args, AE_model)
            writer.add_scalar('eval/Loss', e_loss, step)
            writer.add_scalar('eval/BLER', e_BLER, step)
            writer.add_scalar('eval/BER', e_BER, step)
            writer.add_scalar('eval/RLD', e_RLD, step)
            
            logger.info("[eval] Step: {}/{} ({:.0f}%)\tLoss: {:.6f}\t BER: {:.6f}\t RLD: {:.6f}\t BLER: {:.6f}"
                        .format(step, args.steps, step/args.steps*100, e_loss, e_BER, e_RLD, e_BLER))
            
            if e_BER < best_BER and args.save_model:
                logger.info("saving model...")
                best_BER = e_BER
                checkpoint = {
                    'args': args.__dict__,
                    'encoder_name': AE_model.encoder.__class__.__name__,
                    'encoder_state': AE_model.encoder.state_dict(),
                    'decoder_name': AE_model.decoder.__class__.__name__,
                    'decoder_state': AE_model.decoder.state_dict(),
                    'step': step,
                    'loss': e_loss,
                    'BER': e_BER,
                    'RLD': e_RLD,
                    'BLER': e_BLER,
                    }
                checkpoint_file = os.path.join(log_dir, "checkpoint.pt")
                torch.save(checkpoint, checkpoint_file)
                logger.info("model saved to {}".format(checkpoint_file))
            AE_model.train()
                

def test(args, AE_model):
    pad_token = args.alphabet_size
    bos_token = args.alphabet_size + 1
    vocab_size = args.alphabet_size + 2
    criterion = nn.CrossEntropyLoss()
    
    AE_model.eval()
    
    losses, BERs, BLERs, RLDs = [], [], [], []
    
    with torch.no_grad():
        for step in range(args.eval_size // args.batch_size):
            message = torch.randint(0, args.alphabet_size, (args.batch_size, args.message_length), device=device)
              
            output = AE_model.predict(message).contiguous()
     
            predictions = output.argmax(-1)
            
            RLD = torch.mean(editDistance(predictions, message, None).float()) / message.size(1)
            BLER = torch.mean(torch.any(predictions != message, dim=1).float())
            BER = torch.mean((predictions != message).float())
            
            output_dim = output.shape[-1]
            
            loss = criterion(output.contiguous().view(-1, output_dim), message.view(-1))
            
            losses.append(loss.item())
            BLERs.append(BLER.cpu())
            BERs.append(BER.cpu())
            RLDs.append(RLD.cpu())

    return np.mean(losses), np.mean(BERs), np.mean(BLERs), np.mean(RLDs)
            

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

    channel = lambda x: channels.binaryDeletionChannel(channels.AWGN(x, args.SNR)[0], args.channel_prob)
    
    AE_model = autoencoders.CvtCvtAE(args, channel).to(device)
    # AE_model = autoencoders.ConvConvAE(args, channel).to(device)
    # AE_model = autoencoders.ConvTransformerAE(args, channel).to(device)
    # AE_model = autoencoders.ConvCvtAE(args, channel).to(device)
    
    encoder = AE_model.encoder
    decoder = AE_model.decoder

    # encoder.load_state_dict(torch.load('../runs/Cvt encoder Cvt decoder AWGN 100 to 200 SNR 6 deletion 0/checkpoint.pt')['encoder_state'])
    # decoder.load_state_dict(torch.load('../runs/Cvt encoder Cvt decoder AWGN 100 to 200 SNR 6 deletion 0/checkpoint.pt')['decoder_state'])
            
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
    train(args, AE_model, E_optimizer, D_optimizer)
    test(args, AE_model)


if __name__ == '__main__':
    writer = SummaryWriter(os.path.join(
        log_dir,
        "tensorboard"
    ))
    main(args)