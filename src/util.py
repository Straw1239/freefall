import torch

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



