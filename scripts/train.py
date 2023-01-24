# -*- coding: utf-8 -*-
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


torch.backends.cudnn.benchmark = True

device = torch.device("cuda")

parser = argparse.ArgumentParser(description='Deep Deletion Code')

# General
parser.add_argument('--alphabet_size', type=int, default=3,
                    help='Size of the code alphabet (default: 2)')
parser.add_argument('--code_length', type=int, default=128,
                    help='Length of deletion code (default: 128)')
parser.add_argument('--message_length', type=int, default=64,
                    help='Length message (default: 64)')
parser.add_argument('--deletion_prob', type=float, default=0.1,
                    help='Deletion probability of deletion channel (default: 0.1)')
parser.add_argument('--log_name', type=str, default="DDC_train",
                    help='Name of the log file (default: DDC_train)')

# parser.add_argument('--transition_prob', type=float, default=0.11, metavar='N',
#                     help='Cross transition probability of markov code (default: 0.11)')


# Training args
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training (default: 64)')
parser.add_argument('--steps', type=int, default=4000,
                    help='number of epochs to train (default: 4000)')
parser.add_argument('--gamma', type=float, default=0.7,
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--codeword_sample', type=int, default=16,
                    help='Number of samples for encoder output distribution of codewords (default: 16)')


# Testing args
parser.add_argument('--test_size', type=int, default=1000,
                    help='total samples to test (default: 1000)')
parser.add_argument('--dry_run', action='store_true', default=False,
                    help='quickly check a single pass')

# Encoder args
parser.add_argument('--encoder_lr', type=float, default=0.001, metavar='ELR',
                    help='learning rate (default: 0.001)')
# parser.add_argument('--encoder_hidden', type=int, default=128,
#                     help='hidden layer dimension of encoder (default: 128)')

# Decoder args
parser.add_argument('--decoder_lr', type=float, default=0.001, metavar='DLR',
                    help='learning rate (default: 0.001)')



# Output
parser.add_argument('--checkpoint_name', type=str, default='DDC code',
                    help="name of saved checkpoint (default: DCC code)")
parser.add_argument('--log_interval', type=int, default=10,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_model', action='store_true', default=False,
                    help='For Saving the current Model')

# Misc
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
file_handler = logging.FileHandler('../runs/logs/{}.log'.format(args.log_name), 'w')
stdout_handler = logging.StreamHandler(stream=sys.stdout)
logging.basicConfig(level=logging.INFO, format=FORMAT, handlers=[file_handler, stdout_handler])
logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.linear = nn.Conv1d(args.message_length, args.code_length, kernel_size=1, padding=0)
        self.lstm = nn.LSTM(args.alphabet_size, args.alphabet_size, batch_first=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x has size (batch, message length, alphabet size)
        hidden = self.linear(x)
        logits, _ = self.lstm(hidden)
        # output size is (batch, code length, alphabet size)
        output = self.softmax(logits) 
        return output


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        # self.linear = nn.Conv1d(args.message_length, args.code_length, kernel_size=1, padding=0)
        self.transformer = nn.Transformer(d_model=args.alphabet_size,
                                          dim_feedforward=4*args.alphabet_size,
                                          nhead=8,
                                          num_encoder_layers=6,
                                          num_decoder_layers=6,
                                          batch_first=True)


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, encoder, decoder, E_optimizer, D_optimizer):
    encoder.train()
    decoder.train()
    for step in enumerate(range(args.steps)):
        E_optimizer.zero_grad()
        D_optimizer.zero_grad()
        
        message_bits = torch.randint(0, args.alphabet_size, (args.batch_size, args.message_length), device=device)
        message = F.one_hot(message_bits, args.alphabet_size).float()
        codeword_dist = encoder(message)
        codeword_samples = [codeword_dist[i].multinomial(num_samples=args.codeword_sample, replacement=True).transpose(0, 1) for i in range(args.batch_size)]
        codeword_samples = torch.cat(codeword_samples)
        codeword = F.one_hot(codeword_samples, args.alphabet_size).float()
        pass
        # output = model(data)
        # loss = F.nll_loss(output, target) 
        # loss.backward()
        # optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))
        #     if args.dry_run:
        #         break


def test(args, test_loader, model):
    pass
    # model.eval()
    # test_loss = 0
    # correct = 0
    # with torch.no_grad():
    #     for data, target in test_loader:
    #         data, target = data.to(device), target.to(device)
    #         output = model(data)
    #         test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
    #         pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    #         correct += pred.eq(target.view_as(pred)).sum().item()

    # test_loss /= len(test_loader.dataset)

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))


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
    #     deletion_p=args.deletion_prob,
    #     transition_p=args.transition_prob,
    #     batch_size=args.batch_size)
    
    # test_dataset, test_loader = data_loader(
    #     dataset_size=args.test_size,
    #     length=args.length,
    #     deletion_p=args.deletion_prob,
    #     transition_p=args.transition_prob,
    #     batch_size=args.test_size) 

    encoder = Encoder(args).to(device)
    decoder = Decoder(args).to(device)
    E_optimizer = optim.AdamW(encoder.parameters(), lr=args.encoder_lr)
    D_optimizer = optim.AdamW(decoder.parameters(), lr=args.decoder_lr)
    train(args, encoder, decoder, E_optimizer, D_optimizer)
    



if __name__ == '__main__':
    log_dir = "../runs"
    writer_train = SummaryWriter(os.path.join(
        log_dir,
        args.checkpoint_name,
        "train"
    ))
    writer_val = SummaryWriter(os.path.join(
        log_dir,
        args.checkpoint_name,
        "val"
    ))
    main(args)