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

## load mnist dataset
use_cuda = torch.cuda.is_available()

root = '../data'
download = False  # download MNIST dataset or not

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
test_set = dset.MNIST(root=root, train=False, transform=trans)

batch_size = 64

train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=1,
    shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False)

print('==>>> total trainning batch number: {}'.format(len(train_loader)))
print('==>>> total testing batch number: {}'.format(len(test_loader)))


## network
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 10)
        #self.fc2 = nn.Linear(500, 256)
        #self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        #x = F.selu(self.fc2(x))
        #x = self.fc3(x)
        return x

    def name(self):
        return "MLP"



class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 40, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 40, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.selu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.selu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 40)
        x = F.selu(self.fc1(x))
        x = self.fc2(x)
        return x

    def name(self):
        return "LeNet"


## training
model = LeNet()

celoss = nn.CrossEntropyLoss()

def loss(batch):
    x = torch.cat([x[0] for x in batch], 0)
    target = torch.cat([x[1] for x in batch], 0)
    out = model(x)
    loss = celoss(out, target)
    return loss

def loss_grad(batch):
    x = torch.cat([x[0] for x in batch], 0)
    target = torch.cat([x[1] for x in batch], 0)
    out = model(x)
    loss = celoss(out, target)
    for p in model.parameters():
        loss = loss + 0.0 * torch.norm(p)
    loss.backward()
    result = [g.grad.clone() for g in model.parameters()]
    for p in model.parameters():
        p.grad.zero_()
    return [loss], result

import optimize

dataset = [x for x in train_loader]

optimize.optimize(model.parameters(), loss_grad, optimize.batches(dataset), iterations=2000, grad_batch_split=1, repeat_fraction=0.250, repeat_split=1)


print(loss(dataset))


torch.save(model.state_dict(), model.name())
