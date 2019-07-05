import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import srprop
import numpy as np
import math


import argparse as ap

parser = ap.ArgumentParser("Test BN and GBN on image classification datasets")
parser.add_argument('--norm', help='{NONE, BN, IBN}', metavar='norm')
parser.add_argument('--data', help='{MNIST, CIFAR10, CIFAR100, IMAGENET}', metavar='data', default='CIFAR10')
parser.add_argument('--gpu', metavar='gpu', type=int, default=0)
parser.add_argument('--epochs', metavar='epochs', type=int, default=100)
parser.add_argument('--net', metavar='net', default='convnet')
parser.add_argument('--lr', metavar='lr', type=float, default=0.0005)
parser.add_argument('--cgrad', metavar='cgrad', type=bool, default=False)
parser.add_argument('--iters', metavar='iters', type=int, default=5)
parser.add_argument('--use_prev', metavar='use_prev', type=float, default=0)
parser.add_argument('--cov_roll', metavar='cov_roll', type=float, default=0.9)
parser.add_argument('--w_roll', metavar='w_roll', type=float, default=0.0)

args = parser.parse_args()

device = torch.device('cuda:' + str(args.gpu))

use_cuda = True


datasets = {'MNIST' : dset.MNIST, 'CIFAR10' : dset.CIFAR10, 'CIFAR100' : dset.CIFAR100}
train_loader, test_loader = get_data(datasets[args.data])



from normalization import select_BN
def conv(in, out, ker, stride):
    return select_BN(nn.Conv2D(in, out, ker, stride), args)


model = get_LeNet(conv)

if use_cuda:
    model = model.to(device)

from train import train_model
            
            
            
train_model(model, args.epochs, train_loader, device)     
        
