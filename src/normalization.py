import torch.nn.functional as F
import torch
import torch.nn as nn

import torch.autograd

def standard_BN(conv, *args, **kwargs):
    return nn.Sequential(conv, nn.BatchNorm2d(conv.out_channels))

def no_BN(conv, *args, **kwargs):
    return conv

def i_BN(conv, wroll, covroll,iters, correct_grad, use_prev):
    return IConvBN(conv, wroll, covroll, iters, correct_grad, use_prev)

def select_BN(conv, args):
    types = {'NONE' : no_BN, 'BN' : standard_BN, 'IBN' : i_BN}
    return types[args.norm](conv, args.w_roll, args.cov_roll, correct_grad=args.cgrad, iters=args.iters, use_prev=args.use_prev)

def whitener(x, iters=5, p=None):
    dim = len(x)
    norm = torch.norm(x)
    x = x / norm
    if p is None:
        p = torch.eye(dim, device=x.device)
    for i in range(iters):
        p = 0.5 * (3*p - torch.mm(p, torch.mm(p, torch.mm(p, x))))
        #print(torch.norm(torch.mm(torch.mm(p, x), torch.t(p)) - torch.eye(dim, device=x.device)))
    return p / torch.sqrt(norm)


class IConvBN(nn.Module):

    def __init__(self, conv, wroll=0.9, cov_roll=0.9, iters=7, correct_grad=False, use_prev=False):
        super(IConvBN, self).__init__()
        inputs = conv.in_channels
        self.register_buffer("linear", torch.eye(inputs, requires_grad=False))
        self.register_buffer("bias", torch.zeros(inputs, requires_grad=False))
        self.inputs = inputs
        self.conv = conv
        self.iters = iters
        self.register_buffer("eye", torch.eye(self.inputs))
        self.wroll = wroll
        self.cov_roll = cov_roll
        self.register_buffer("cov", torch.eye(self.inputs))
        self.eps = 1e-3
        self.correct_grad = correct_grad
        self.use_prev= use_prev
        

        
    def fused_apply(self, x):
        linear = self.linear
        bias = self.bias
        if not self.correct_grad:
            linear = linear.detach()
            bias = bias.detach()
        weight = torch.matmul(linear, self.conv.weight.permute(2, 3, 1, 0)).permute(3, 2, 0, 1)
        bias = torch.matmul(self.conv.weight.sum((2, 3)), bias) + self.conv.bias
        return F.conv2d(x, weight, bias, self.conv.stride, self.conv.padding, self.conv.dilation, 1)
        
        
    def update_whitener(self, x):
        grad_gate = torch.enable_grad() if self.correct_grad else torch.no_grad()

        with grad_gate:
            x = x.view(-1, self.inputs)
            sample_mean = torch.mean(x, 0)
            centered_x = x - sample_mean
            sample_cov = torch.matmul(torch.t(centered_x), centered_x) / x.shape[0]
            self.bias = self.bias.detach()
            self.linear = self.linear.detach()
            self.cov = self.cov.detach()
            self.cov *= self.cov_roll
            self.bias *= 0.9
            self.linear *= self.wroll
        

            
            self.cov += (1 - self.cov_roll) * sample_cov
            self.bias -= (0.1) * sample_mean
            guess = self.use_prev * self.linear + (1-self.use_prev) * self.eye 
            self.linear += (1-self.wroll) * whitener(self.cov + self.eps * self.eye, self.iters, guess)

    def forward(self, x):
        self.update_whitener(x)
        return self.fused_apply(x)

    




