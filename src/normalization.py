import torch.nn.functional as F
import torch
import torch.nn as nn

import torch.autograd

class GenConvBN(nn.Module):

    def __init__(self, conv, iters=1):
        super(GenConvBN, self).__init__()
        inputs = conv.in_channels
        self.linear = torch.eye(inputs, requires_grad=False)
        self.bias = torch.zeros(inputs, requires_grad=False)
        self.inputs = inputs
        self.conv = conv
        self.iters = iters

    def fused_apply(self, x):
        weight = torch.matmul(self.linear.detach(), self.conv.weight.permute(2, 3, 1, 0)).permute(3, 2, 0, 1)
        bias = torch.matmul(self.conv.weight.sum((2, 3)), self.bias.detach()) + self.conv.bias
        return F.conv2d(x, weight, bias, self.conv.stride, self.conv.padding, self.conv.dilation, 1)
        #x = F.conv2d(x, self.linear.unsqueeze(-1).unsqueeze(-1), self.bias)
        #return self.conv.forward(x)

    def calc_grad(self, mean, cov):
        transformed_cov = torch.matmul(self.linear, torch.matmul(cov, torch.t(self.linear)))
        cov_correction = transformed_cov - torch.eye(self.inputs)
        cov_loss = cov_correction.norm()

        cov_grad = 2 / cov_loss * torch.matmul(torch.matmul(cov_correction, self.linear), cov)

        mean_grad = 2 * (self.bias - mean)
        #print((self.bias - mean).norm(), cov_loss)
        return mean_grad, cov_grad

    def optimize(self, x, lr=0.1, iters=1):
        with torch.no_grad():
            x = x.view(-1, self.inputs)
            sample_mean = torch.mean(x, 0)
            centered_x = x - sample_mean
            sample_cov = torch.matmul(torch.t(centered_x), centered_x) / x.shape[0]

            for i in range(iters):
                bias_grad, linear_grad = self.calc_grad(sample_mean, sample_cov)
                self.bias -= lr * bias_grad
                self.linear -= lr * linear_grad

    def forward(self, x, lr=0.2):
        self.optimize(x, lr, self.iters)
        return self.fused_apply(x)



