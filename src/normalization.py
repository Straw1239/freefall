import torch.nn.functional as F
import torch
import torch.nn as nn

import torch.autograd


class GenConvBN(nn.Module):

    def __init__(self, conv, roll=0, iters=1):
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
        self.eps = 1e-2
        

        

    def fused_apply(self, x):
        weight = torch.matmul(self.linear, self.conv.weight.permute(2, 3, 1, 0)).permute(3, 2, 0, 1)
        bias = torch.matmul(self.conv.weight.sum((2, 3)), self.bias) + self.conv.bias
        return F.conv2d(x, weight, bias, self.conv.stride, self.conv.padding, self.conv.dilation, 1)
        #x = F.conv2d(x, self.linear.unsqueeze(-1).unsqueeze(-1), self.bias)
        #return self.conv.forward(x)

    def calc_grad(self, mean, cov):
        transformed_cov = torch.matmul(self.linear, torch.matmul(cov, torch.t(self.linear)))
        cov_correction = transformed_cov - self.eye
        cov_loss = cov_correction.norm()
        cov_grad = 2 / (cov_loss + self.eps) * torch.matmul(torch.matmul(cov_correction, self.linear), cov)

        mean_grad = 2 * (self.bias - mean)
        #print((self.bias - mean).norm(), cov_loss)
        return mean_grad, cov_grad

    def optimize(self, x, lr, iters=1):
        biaslr= 0.1
        x = x.view(-1, self.inputs)
        sample_mean = torch.mean(x, 0)
        centered_x = x - sample_mean
        sample_cov = torch.matmul(torch.t(centered_x), centered_x) / x.shape[0]
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
        self.optimize(x, lr, self.iters)
        #print(torch.max(torch.abs(self.linear)), torch.max(torch.abs(self.bias)))
        return self.fused_apply(x)



