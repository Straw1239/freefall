import torch
import IsoSGD
import numpy as np

import operator
from torch.autograd import Variable



from itertools import accumulate
def split_batch(batch, fractions):
    split_fracs = list(accumulate(fractions))
    split_fracs = [int(round(len(batch) * x)) for x in split_fracs]
    results = [batch[0:split_fracs[1]]]
    for i in range(len(split_fracs) - 1):
        results.append(batch[split_fracs[i]:split_fracs[i+1]])
    return results

def list_trans(lists):
    return list(map(list, zip(*lists)))

def suml(lists):
    tranpose = list_trans(lists)
    result = []
    for component in tranpose:
        result.append(torch.sum(torch.stack(component, 0), 0))
    return result

def apply(lists, f):
    return [f(*args) for args in list_trans(lists)]

def mean_var(grads, weights=None):
    if weights is None:
        weights = [[1.0 / len(grads) for _ in g] for g in grads]
    mean = suml([apply([g, w], operator.mul) for g, w in zip(grads, weights)])

    def f(g, w):
        return g * g * w

    def square(x):
        return x * x

    moment = suml([apply([g, w], f) for g, w in zip(grads, weights)])
    var = apply([moment, apply([mean], square)], operator.sub)
    for v in var:
        v.mul_(1.0 / (len(grads) - 1))
    return mean, var



def loss_grad_mean_var(batches, loss_and_grad):
    loss_grads = [loss_and_grad(b) for b in batches]
    return mean_var([x for x, _ in loss_grads]), mean_var([x for _, x in loss_grads])




def optimize(parameters, loss_and_grad, batch_source, iterations=10000, grad_batch_split=2, repeat_fraction=0.2, repeat_split=2):
    opt = IsoSGD.IsoSGD(parameters)
    batch = next(batch_source)
    over_mini_size = repeat_fraction / repeat_split
    main_mini_size = (1 - repeat_fraction) / grad_batch_split
    splits = [over_mini_size for _ in range(repeat_split)] + [main_mini_size for _ in range(grad_batch_split)]
    batches = split_batch(batch, splits)


    stored_grads = []
    stored_losses = []
    for b in batches[:repeat_split]:
        loss, grad = loss_and_grad(b)
        stored_grads.append(grad)
        stored_losses.append(loss)

    opt.zero_grad()

    for i in range(iterations):
        batch = next(batch_source)
        overlap = batches[:repeat_split]
        batches = split_batch(batch, splits)
        loss_grads = [loss_and_grad(b) for b in batches[repeat_split:]]
        grads = [x for _, x in loss_grads]
        losses = [x[0].item() for x, _ in loss_grads]
        loss = sum(losses) / len(losses)
        print(loss)
        overlap_loss_grad = [loss_and_grad(b) for b in overlap]
        overlap_grad = [x for _, x in overlap_loss_grad]
        loss_overlap = [x for x, _ in overlap_loss_grad]
        loss_deltas = [[nc - oc for nc, oc in zip(n, o)] for n, o in zip(loss_overlap, stored_losses)]

        deltas = [[nc - oc for nc, oc in zip(n, o)] for n, o in zip(overlap_grad, stored_grads)]
        stored_grads = []
        stored_losses = []
        for b in batches[:repeat_split]:
            loss, grad = loss_and_grad(b)
            stored_grads.append(grad)
            stored_losses.append(loss)


        opt.step(grads, deltas)
        opt.zero_grad()

def batches(dataset, batch_size=64):
    dataset = np.array(dataset, dtype=np.object)
    while True:
        i = 0
        while i < (len(dataset) - batch_size):
            yield dataset[i: i + batch_size]
            i += batch_size
        np.random.shuffle(dataset)


def main():

    #scale = torch.diag(torch.cat(100))
    dim = 10
    x = torch.randn([dim], requires_grad=True)
    x_mutable = x.detach()
    #x_mutable *= 100
    N = 1000
    c = [torch.randn(dim, dim) / 500 for _ in range(N)]
    l = [torch.randn(dim) / 1000 for _ in range(N)]

    ltarget = torch.zeros(dim)

    lcorrection = (torch.sum(torch.stack(l, 0), 0) - ltarget) / N

    l = [x - lcorrection for x in l]

    target = torch.randn(dim, dim) / 1
    target = torch.mm(target, torch.t(target))
    target += torch.eye(dim)
    correction = (torch.sum(torch.stack(c, 0), 0) - target) / N
    c = [x - correction for x in c]






    def loss_grad(batch):
        quad_form = torch.sum(torch.stack([q for q, _ in batch], 0), 0) * (N / len(batch))
        linear = torch.sum(torch.stack([lt for _, lt in batch], 0), 0)
        loss = torch.dot(torch.mv(quad_form, x), x)
        loss = loss + torch.dot(x, linear)
        loss.backward()
        result = [x.grad.clone()]
        x.grad.zero_()
        return [loss], result

    batch_gen = batches(list(zip(c, l)), 64)

    optimize([x], loss_grad,  batch_gen, iterations=10000, grad_batch_split=4, repeat_fraction=0.5, repeat_split=8)

if __name__ == '__main__':
    main()