import torch
import srprop
import numpy as np
from torch.autograd import Variable

x = Variable(-torch.ones([2]), requires_grad=True)
c = Variable(torch.from_numpy(np.array([1, 1])).type(torch.FloatTensor))



optimizer = srprop.SRProp([x], 0)
noise_mean = torch.zeros_like(x)
noise_std = 1*torch.ones_like(x)
for i in range(1000):
    y = torch.sum(x * x * c)
    torch.autograd.backward(y)
    g = x.grad + torch.normal(noise_mean, noise_std)
    optimizer.step([g], [noise_std*noise_std])
    x.grad.data.zero_()
    #if(y.data[0] < 0.01):
        #print(i)
        #break
    if i % 1 == 0:
        print(y.data[0], end=' ')