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
parser.add_argument('--norm', help='{NONE, BN, GBN-C, GBN-N}', metavar='norm')
parser.add_argument('--data', help='{MNIST, CIFAR10, CIFAR100, IMAGENET}', metavar='data', default='CIFAR10')
parser.add_argument('--gpu', metavar='gpu', type=int, default=0)
parser.add_argument('--epochs', metavar='epochs', type=int, default=100)
parser.add_argument('--net', metavar='net', default='convnet')
parser.add_argument('--lr', metavar='lr', type=float, default=0.0005)


args = parser.parse_args()

device = torch.device('cuda:' + str(args.gpu))

use_cuda = True

root = '../data'
download = False

datasets = {'MNIST' : dset.MNIST, 'CIFAR10' : dset.CIFAR10, 'CIFAR100' : dset.CIFAR100}

trans = transforms.ToTensor()
data_set = datasets[args.data]
train_set = data_set(root=root, train=True, transform=trans, download=download)
test_set = data_set(root=root, train=False, transform=trans)

batch_size = 64

train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True, pin_memory=True, num_workers=8)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False, pin_memory=True, num_workers=8)


from normalization import select_BN

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        
        self.conv1 = select_BN(nn.Conv2d(3, 40, 5, 1), args.norm)
        self.conv2 = select_BN(nn.Conv2d(40, 40, 5, 1), args.norm)
        self.fc1 = nn.Linear(5 * 5 * 40, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        
        x = F.selu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        
        x = F.selu(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5 * 5 * 40)
        x = F.selu(self.fc1(x))
        x = self.fc2(x)
        return x

    def name(self):
        return "LeNet"


## training
model = LeNet()

if use_cuda:
    model = model.to(device)


import time

def standard_reporter(report_interval):
    running_loss = [torch.zeros(1, device=device)]
    start = time.time()
    def reporter(loss, iters, epoch, batch_idx):        
        running_loss[0] += loss
        if iters % report_interval == 0:
            print(time.time() - start, float(running_loss[0].cpu()) / report_interval)
            running_loss[0] *= 0

    return reporter
            
            
            
    
    

def train_model(model, epochs, data_loader, device, optimizer=optim.SGD(model.parameters(), lr=args.lr, momentum=0.9), report=standard_reporter(100)):

    lf = nn.CrossEntropyLoss()
    iters = 0
    for epoch in range(epochs):
        for batch_idx, (x, target) in enumerate(data_loader):
            iters += 1
            optimizer.zero_grad()
            if use_cuda:
                x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True)
                out = model(x)
            
            loss = lf(out, target)
            loss.backward()
            optimizer.step()
            report(loss.detach(), iters, epoch, batch_idx)
            
            
            
train_model(model, args.epochs, train_loader, device)     
        
