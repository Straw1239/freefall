import torch.nn.functional as F
import torch
import torch.nn as nn

import torch.autograd

def standard_BN(conv):
    return nn.Sequential(conv, nn.BatchNorm2d(conv.out_channels))

def gen_BN_no_grad(conv):
    return GenConvBN(conv, correct_grad=False)

def gen_BN(conv):
    return GenConvBN(conv, correct_grad=True)

def no_BN(conv):
    return conv

def i_BN(conv):
    return IConvBN(conv)

def select_BN(conv, norm_type):
    types = {'NONE' : no_BN, 'BN' : standard_BN, 'GBN-C' : gen_BN, 'GBN-N': gen_BN_no_grad, 'IBN' : i_BN}
    return types[norm_type](conv)

def whitener(x, iters=5):
    dim = len(x)
    norm = torch.norm(x)
    x = x / norm
    p = torch.eye(dim, device=x.device)
    for i in range(iters):
        p = 0.5 * (3*p - torch.mm(p, torch.mm(torch.t(p), torch.mm(p, x))))
        #print(torch.norm(torch.mm(torch.mm(p, x), torch.t(p)) - torch.eye(dim, device=x.device)))
    return p / torch.sqrt(norm)


class IConvBN(nn.Module):

    def __init__(self, conv, roll=0.9, iters=7, correct_grad=False):
        super(IConvBN, self).__init__()
        inputs = conv.in_channels
        self.register_buffer("linear", torch.eye(inputs, requires_grad=False))
        self.register_buffer("bias", torch.zeros(inputs, requires_grad=False))
        self.inputs = inputs
        self.conv = conv
        self.iters = iters
        self.register_buffer("eye", torch.eye(self.inputs))
        self.roll = roll
        self.register_buffer("cov", torch.eye(self.inputs))
        self.eps = 1e-3
        self.correct_grad = correct_grad
        

        
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

            self.bias *= self.roll
            self.linear *= self.roll
            update = 1 - self.roll

            self.bias -= update * sample_mean
            self.linear += update * whitener(sample_cov + self.eps * self.eye, self.iters)

            
        

   

    def forward(self, x):
        self.update_whitener(x)
        return self.fused_apply(x)

    

class GenConvBN(nn.Module):

    def __init__(self, conv, roll=0, iters=1, correct_grad=False):
        super(GenConvBN, self).__init__()
        inputs = conv.in_channels
        self.register_buffer("linear", torch.eye(inputs, requires_grad=False))
        self.register_buffer("bias", torch.zeros(inputs, requires_grad=False))
        self.inputs = inputs
        self.conv = conv
        self.iters = iters
        self.register_buffer("eye", torch.eye(self.inputs))
        self.roll = roll
        self.register_buffer("cov", torch.eye(self.inputs))
        self.eps = 1e-3
        self.correct_grad = correct_grad
        self.count = 0
        

        
    def fused_apply(self, x):
        linear = self.linear
        bias = self.bias
        if not self.correct_grad:
            linear = linear.detach()
            bias = bias.detach()
        weight = torch.matmul(linear, self.conv.weight.permute(2, 3, 1, 0)).permute(3, 2, 0, 1)
        bias = torch.matmul(self.conv.weight.sum((2, 3)), bias) + self.conv.bias
        return F.conv2d(x, weight, bias, self.conv.stride, self.conv.padding, self.conv.dilation, 1)
        
        

    def calc_grad(self, mean, cov):
        transformed_cov = torch.matmul(self.linear, torch.matmul(cov, torch.t(self.linear)))
        cov_correction = transformed_cov - self.eye
        cov_loss = cov_correction.norm()
        cov_grad = 2 / (cov_loss + self.eps) * torch.matmul(torch.matmul(cov_correction, self.linear), cov)

        mean_grad = 2 * (self.bias - mean)
        #print((self.bias - mean).norm(), cov_loss)
        return mean_grad, cov_grad

    def optimize(self, x, lr, iters=1):

        grad_gate = torch.enable_grad() if self.correct_grad else torch.no_grad()

        with grad_gate:
            biaslr= 0.1
            x = x.view(-1, self.inputs)
            sample_mean = torch.mean(x, 0)
            centered_x = x - sample_mean
            sample_cov = torch.matmul(torch.t(centered_x), centered_x) / x.shape[0] + self.eps * self.eye
            self.cov = self.cov.detach()
            self.cov *= self.roll
            self.cov += (1 - self.roll) + sample_cov
            self.linear = self.linear.detach()
            self.bias = self.bias.detach()
            for i in range(iters):
                bias_grad, linear_grad = self.calc_grad(sample_mean, self.cov + self.eps * self.eye)
                self.bias = self.bias - biaslr * bias_grad
                self.linear = self.linear - lr * linear_grad

    def forward(self, x, lr=0.001):
        if self.count == 0:
            self.optimize(x, lr, 3000)
        else:
            self.optimize(x, lr, self.iters)
        self.count += 1
        #print(torch.max(torch.abs(self.linear)), torch.max(torch.abs(self.bias)))
        return self.fused_apply(x)



