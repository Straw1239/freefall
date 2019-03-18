import torch

from filter import kalman_update
import math

def normalcdf(x, mu, sigma):
    z = (x - mu) / (math.sqrt(2) * sigma)
    z.data.clamp_(-5, 5)
    return 1.0 / 2 * (1 + torch.erf(z))

def r_update(param, new, forget=0.95):
    param *= forget
    param += new * (1 - forget)

def square(x):
    return x * x

def tdot(x, y):
    return torch.dot(x.view(-1), y.view(-1))

def tnorm2(x):
    return tdot(x, x)

def ddot(x, y):
    return sum(tdot(a, b) for a, b in zip(x, y))

def dnorm2(x):
    return ddot(x, x)

def dnorm(x):
    return torch.sqrt(dnorm2(x))

class DirectSum:
    def __init__(self, tensors):
        self.tensors = tensors

    def __add__(self, other):
        return DirectSum([x + y for x, y in zip(self, other)])

    def __iter__(self):
        return self.tensors.__iter__()

    def __sub__(self, other):
        return DirectSum([x - y for x, y in zip(self, other)])

    def __iadd__(self, other):
        for x, y in zip(self, other):
            x += y
        return self

    def __isub__(self, other):
        for x, y in zip(self, other):
            x -= y
        return self

    def __imul__(self, other):
        for x in self:
            x *= other
        return self

    def __mul__(self, other):
        return DirectSum([x * other for x in self])

    def __str__(self):
        return self.tensors.__str__()

    def __neg__(self):
        return DirectSum([-x for x in self])

    def view(self, *args):
        return DirectSum([x.view(*args) for x in self])

    def apply(self, func):
        self.tensors = [func(x) for x in self.tensors]


class IsoSGD:
    def __init__(self, params):
        self.params = list(params)
        #self.grad_var = torch.ones(1)
        #self.grad_mean = DirectSum([torch.zeros_like(p) for p in self.params])
        #self.curvature = torch.ones(1)
        #self.obs_noise = torch.ones(1)
        #self.process_noise = torch.ones(1) * 0.1
        #self.grad_delta_cov = torch.zeros(1)
        #self.grad_norm2 = torch.ones(1)
        self.last_step = DirectSum([torch.ones_like(p) for p in self.params])
        #self.total_dim = sum(p.numel() for p in self.params)
        self.move_bound = 0.01 * torch.ones(1)
        self.avg_dir = DirectSum([torch.zeros_like(p) for p in self.params])
        self.avg_curv = DirectSum([torch.zeros_like(p) for p in self.params])



    def step(self, grads, deltas):
        grads = [DirectSum(x) for x in grads]
        deltas = [DirectSum(x) for x in deltas]

        grad_mean = sum(grads[1:], grads[0]) * (1.0 / len(grads))
        delta_mean = sum(deltas[1:], deltas[0]) * (1.0 / len(deltas))

        #Incorrect?
        #grad_var = sum([dnorm2(x - grad_mean) for x in grads]) * (1.0 / (len(grads)*(len(grads) - 1)))
        #delta_var = sum([dnorm2(x - delta_mean) for x in deltas]) * (1.0 / (len(deltas) * (len(deltas) - 1)))
        #r_update(self.obs_noise, grad_var)
        #r_update(self.process_noise, torch.clamp(dnorm2(delta_mean) - delta_var, 1e-15, 1e15), forget=0.9)
        #Something wrong with this curv measure-should be fixed now
        #r_update(self.curvature, -ddot(delta_mean, self.last_step) / dnorm2(self.last_step), forget=0.9)
        #print((-ddot(delta_mean, self.last_step) / dnorm2(self.last_step).item()))
        #r_update(self.grad_delta_cov, ddot(delta_mean, grad_mean))
        #r_update(self.grad_norm2, torch.clamp(dnorm2(grad_mean) - grad_var, 0, 1e10))


        #est_decay = self.grad_delta_cov / torch.sqrt(self.grad_norm2)
        #est_decay.clamp_(-2, 0)
        #est_decay = 0

        #kalman_update(self.grad_mean, self.grad_var, self.process_noise, self.obs_noise, grad_mean, 1 + est_decay)

        #move_frac = -(2 * normalcdf(0, dnorm(grad_mean), torch.sqrt(grad_var)) - 1)


        lr = 0.1


        r_update(self.avg_dir, self.last_step * (1.0 / dnorm(self.last_step)), 0.3)
        r_update(self.avg_curv, delta_mean * (1.0 / dnorm(self.last_step)), 0.3)

        self.avg_dir *= 1.0 / dnorm(self.avg_dir)
        curv = -ddot(self.avg_dir, self.avg_curv)
        step = grad_mean * lr

        boost = 0
        grad_curve_info = ddot(grad_mean, self.avg_dir)
        if curv > 0:
            #Sherman-Morrison Inverse
            #b_len = dnorm2()
            v = self.avg_dir
            u = self.avg_curv - (self.avg_dir * (1.0 / lr))
            boost = -u * grad_curve_info * (lr * lr / (1 + lr * ddot(u, v)))
            boost_len = dnorm(boost)
            boost *= 1.0 / boost_len
            boost *= min(boost_len, self.move_bound)
        else:
            boost = self.avg_dir * ((self.move_bound / dnorm(self.avg_dir)) * torch.sign(grad_curve_info))

        step += boost



        print(curv, dnorm(step), self.move_bound, end=' ')
        for p, s in zip(self.params, step):
            p.detach().sub_(s)
        self.last_step = step
        r_update(self.move_bound, dnorm(boost) * 1.1, 0.8)
        #print(self.curvature * torch.sqrt(self.grad_var), self.process_noise, est_decay)



    def zero_grad(self):
        for p in self.params:
            p.grad.zero_()
