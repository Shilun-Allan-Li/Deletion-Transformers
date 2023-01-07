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

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

device = torch.device("cuda")

parser = argparse.ArgumentParser(description='Deep Deletion Code')

# General
parser.add_argument('--code_length', type=int, default=128, metavar='N',
                    help='Length of deletion code (default: 128)')
parser.add_argument('--message_length', type=int, default=64, metavar='N',
                    help='Length message (default: 64)')
parser.add_argument('--deletion_prob', type=float, default=0.1, metavar='N',
                    help='Deletion probability of deletion channel (default: 0.1)')
# parser.add_argument('--transition_prob', type=float, default=0.11, metavar='N',
#                     help='Cross transition probability of markov code (default: 0.11)')


# Training args
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--steps', type=int, default=4000, metavar='N',
                    help='number of epochs to train (default: 4000)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')


# Testing args
parser.add_argument('--test_size', type=int, default=1000, metavar='N',
                    help='total samples to test (default: 1000)')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')

# Encoder args


# Decoder args



# Output
parser.add_argument('--checkpoint_name', type=str, default='DDC code')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')

# Misc
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')


class Encoder(nn.Module):
    def __init__(self, message_length, code_length):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(1, 1)

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
    
class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

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


def train(args, train_loader, test_loader, model, optimizer):
    model.train()
    for step, (x, y, deletion_mask, padding_mask) in enumerate(train_loader):
        x, y, padding_mask = x.to(device), y.to(device), padding_mask.to(device)
        optimizer.zero_grad()
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

    # model = Net(args).to(device)
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr)



if __name__ == '__main__':
    args = parser.parse_args()
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