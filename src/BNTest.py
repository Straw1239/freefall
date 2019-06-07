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

device = torch.device('cuda:0')
## load mnist dataset
use_cuda = torch.cuda.is_available()

root = '../data'
download = True  # download MNIST dataset or not

trans = transforms.ToTensor()
train_set = dset.CIFAR10(root=root, train=True, transform=trans, download=download)
test_set = dset.CIFAR10(root=root, train=False, transform=trans)

batch_size = 64

train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True, pin_memory=True, num_workers=8)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False, pin_memory=True, num_workers=8)

#print('==>>> total trainning batch number: {}'.format(len(train_loader)))
#print('==>>> total testing batch number: {}'.format(len(test_loader)))


from normalization import GenConvBN

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.BN0 = nn.BatchNorm2d(3)
        self.conv1 =  GenConvBN(nn.Conv2d(3, 40, 5, 1))
        self.BN2 = nn.BatchNorm2d(40)
        self.conv2 = GenConvBN(nn.Conv2d(40, 40, 5, 1))
        self.fc1 = nn.Linear(5 * 5 * 40, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.BN0(x)
        x = F.selu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        #x = self.BN2(x)
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


import sys
optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)

ceriation = nn.CrossEntropyLoss()
iter = 0

for epoch in range(50):
    # trainning
    ave_loss = torch.zeros(1, device=device)
    for batch_idx, (x, target) in enumerate(train_loader):
        iter += 1
        optimizer.zero_grad()
        if use_cuda:
            x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True)
        if iter == 1:
            model.conv1.iters = 3000
            model.conv2.iters = 3000
        out = model(x)
        if iter == 1:
            model.conv1.iters= 1
            model.conv2.iters = 1
        loss = ceriation(out, target)
        ave_loss = ave_loss * 0.99 + 0.01 * loss.detach()
        loss.backward()
        optimizer.step()
        if iter % 100 == 0:
            #print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(
                #epoch, batch_idx + 1, ave_loss))
            print(float(ave_loss.item()))

    # testing
    """
    correct_cnt, a
    for batch_idx, (x, target) in enumerate(test_loader):
        if use_cuda:
            x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True)
        out = model(x)
        loss = ceriation(out, target)
        _, pred_label = torch.max(out.data, 1)
        total_cnt += x.data.size()[0]
        correct_cnt += (pred_label == target.data).sum()
        # smooth average
        ave_loss = ave_loss * 0.9 + loss.data.item() * 0.1

        #if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(test_loader):
            #print('==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}'.format(
       """         #epoch, batch_idx + 1, ave_loss, correct_cnt * 1.0 / total_cnt))

#torch.save(model.state_dict(), model.name())
