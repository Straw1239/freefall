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
download = True  # download MNIST dataset or not

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
test_set = dset.MNIST(root=root, train=False, transform=trans)

batch_size = 50

train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
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
        self.fc1 = nn.Linear(28 * 28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = self.fc3(x)
        return x

    def name(self):
        return "MLP"


def loss_grad_distribution(mod, loss_batch, grad=None, grad_var=None):

    if grad is None:
        grad = [torch.zeros_like(x) for x in mod.parameters()]
    if grad_var is None:
        grad_var = [torch.zeros_like(x) for x in mod.parameters()]
    loss_mean = 0
    loss_var = 0
    for loss in loss_batch:
        mod.zero_grad()
        torch.autograd.backward(loss, retain_graph=True)
        loss_mean += loss.data[0]
        loss_var += loss.data[0] * loss.data[0]
        for p, g, v in zip(mod.parameters(), grad, grad_var):
            g += p.grad
            v += p.grad * p.grad
    n = len(loss_batch)
    loss_mean /= n
    loss_var /= n
    loss_var -= loss_mean * loss_mean
    loss_var /= (n - 1)
    for g, g2 in zip(grad, grad_var):
        g /= n
        g2 /= n
        g2 -= g*g
        g2 /= (n - 1)
    return (loss_mean, loss_var), (grad, grad_var)




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

if use_cuda:
    model = model.cuda()

optimizer = srprop.SRProp(model.parameters())#optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

celoss = nn.CrossEntropyLoss()

for epoch in range(10):
    # trainning
    ave_loss = 0
    for batch_idx, (x, target) in enumerate(train_loader):
        #optimizer.zero_grad()
        if use_cuda:
            x, target = x.cuda(), target.cuda()
        x, target = Variable(x), Variable(target)
        out = model(x)
        #for o, t in zip(out, target):
            #print(o.shape, t.shape)
        #print(out.shape, target.shape)
        losses = [celoss(out[i:i+1, :], target[i:i+1]) for i in range(batch_size)]
        (loss_mean, loss_var), (grad_mean, grad_var) = loss_grad_distribution(model, losses)
       # print('zvar', [np.any(gv.data.numpy() == 0) for gv in grad_var])
       # print(loss_mean, math.sqrt(loss_var))
        ave_loss = ave_loss * 0.95 + loss_mean * 0.05
        optimizer.step(grad_mean, grad_var)
        if (batch_idx + 1) % 1 == 0 or (batch_idx + 1) == len(train_loader):
            print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(epoch, batch_idx + 1, ave_loss))
    # testing
    correct_cnt, ave_loss = 0, 0
    total_cnt = 0
    for batch_idx, (x, target) in enumerate(test_loader):
        if use_cuda:
            x, targe = x.cuda(), target.cuda()
        x, target = Variable(x, volatile=True), Variable(target, volatile=True)
        out = model(x)
        loss = celoss(out, target)
        _, pred_label = torch.max(out.data, 1)
        total_cnt += x.data.size()[0]
        correct_cnt += (pred_label == target.data).sum()
        # smooth average
        ave_loss = ave_loss * 0.95 + loss.data[0] * 0.05

        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(test_loader):
            print('==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}'.format(
                epoch, batch_idx + 1, ave_loss, correct_cnt * 1.0 / total_cnt))

torch.save(model.state_dict(), model.name())
