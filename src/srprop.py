
import math
import torch
import numpy as np
import collections

def normalcdf(x, mu, sigma):
    z = (x - mu) / (math.sqrt(2) * sigma)
    z.data.clamp_(-5, 5)
    return 1.0 / 2 * (1 + torch.erf(z))


def square(x):
    return x * x


def normal_ratio_pdf(a, b, x):
    q = (b + a * x) / torch.sqrt(1 + square(x))

    integral = q * torch.exp(1.0 / 2 * (square(q) - (square(a) + square(b)))) * math.sqrt(math.pi / 2) * torch.erf(q * math.sqrt(1.0 / 2))
    return (torch.exp(-1.0 / 2 * (square(a) + square(b))) + integral) / (math.pi * (1 + square(x)))



def ratio_integral(uz, uw, sz, sw, p, f, points=51, sstep=0.1):
    s = p * sz / sw
    r = sw / (sz * torch.sqrt(1 - square(p)))
    b = uw / sw
    a = (uz / sz - p * uw / sw) / torch.sqrt(1 - square(p))
    step = sstep / (torch.abs(b) + 1)
    result = torch.zeros_like(uz)
    b[b == 0] = 0.01
    for i in range(points):
        x = (i * step - step * points / 2 + (a / b))
        densities = normal_ratio_pdf(a, b, x)
        evals = f(x / r + s)
        result += densities * evals
    result *= step
    return result

class SRProp(object):
    def __init__(self, params, step=1e-3, multipliers=(0.5, 1.2)):
        self.opts = [SRPropT(p, step, multipliers) for p in params]

    def step(self, grads, grad_vars):
        for opt, g, gv in zip(self.opts, grads, grad_vars):
            opt.step(g, gv)



class SRPropT(object):
    def __init__(self, params, step, multipliers):
        self.params = params
        self.steps = torch.ones_like(params) * step
        (self.n_mult, self.p_mult) = multipliers
        self.grad_mean = torch.zeros_like(params)
        self.grad_var = torch.ones_like(params)
        self.noise_var = 0.2*torch.ones_like(params)

        self.prev_move_frac = torch.ones_like(params)
        self.avg_obs_noise = torch.ones_like(params)



    def step(self, new_grad, overlap_grad, old_grad):
        eps = 1e-4

        cov_falloff = 0.05
        for cov, lag_val in zip(self.auto_covs, self.past_errs):
            cov *= (1 - cov_falloff)
            cov += cov_falloff * (lag_val * err)
        self.avg_obs_noise *= 1 - cov_falloff
        self.avg_obs_noise += cov_falloff * grad_var


        k = 1.0 / (self.grad_var + self.avg_obs_noise + self.noise_var)



        old_grad_mean = self.grad_mean + (self.grad_var * (grad - self.grad_mean)) * k
        self.grad_mean += (self.grad_var + self.noise_var) * (grad - self.grad_mean) * k

        old_grad_var = self.grad_var - k * square(self.grad_var)
        grad_corr = self.grad_var - k * (self.grad_var + self.noise_var) * self.grad_var
        self.grad_var = self.grad_var + self.noise_var - k * square(self.grad_var + self.noise_var)
        grad_sigma = torch.sqrt(self.grad_var)
        old_grad_sigma = torch.sqrt(old_grad_var)
        grad_sigma.data.clamp_(eps, 1e10)
        grad_corr /= (grad_sigma * old_grad_sigma)


        scale = 5
        self.prev_move_frac.abs_()
        m = self.prev_move_frac.data.numpy()
        print('\tmove:', np.mean(m), np.min(m), np.max(m), sep='\t', end=' ')
        self.prev_move_frac.data.clamp_(0.2, 2)
        def log_step_mult(grad_ratio):
            return torch.log(torch.sigmoid(scale * ((grad_ratio - 1) / self.prev_move_frac + 1)) * (self.p_mult - self.n_mult) + self.n_mult)

        step_mults = ratio_integral(self.grad_mean, old_grad_mean, self.grad_var, old_grad_var, grad_corr,
                                    log_step_mult, 51, 0.1)

        step_mults.exp_()
        step_mults = step_mults.clamp(self.n_mult, self.p_mult)
        step_mults[self.prev_move_frac == 0.2] = 1
        #print('corrs:', np.array(grad_corr.data), end=' ')
        dat = self.noise_var.data.numpy()
        print('\tnoise:', np.mean(dat), np.min(dat), np.max(dat), sep='\t', end=' ')
        #print('err:', np.array(err.data), end=' ')
        #print('x:', np.array(self.params.data), end=' ')
        #print('grad', np.array(self.grad_mean.data), end='\n')
        #print('std', np.array(grad_sigma.data))

        s = self.steps.data.numpy()
        print('\tsteps:', np.mean(s), np.min(s), np.max(s), sep='\t')

        self.steps *= step_mults
        self.steps.data.clamp_(1e-8, 10)
        k = self.avg_k
        self.prev_move_frac = (2 * normalcdf(0, self.grad_mean, grad_sigma) - 1)
        self.prev_move_frac.data.clamp_(-1, 1)
        self.params.data += (self.steps * self.prev_move_frac).data

        self.noise_var = self.cov_lags * ((self.auto_covs[0] - self.avg_obs_noise)*(square(k) + 2*k) - square(k) * self.avg_obs_noise)
        for i in range(1, self.cov_lags):
            k2 = k * k
            kim1 = (1 - k) ** (i - 1)
            ki = kim1 * (1 - k)
            self.noise_var += 2*(self.cov_lags - i)*(((self.auto_covs[i] + kim1) * k * self.avg_obs_noise * (k2 + 2*k)) / ki - k2 * self.avg_obs_noise)
        self.noise_var *= 1.0 / (self.cov_lags * self.cov_lags)

        self.noise_var.data.clamp_(1e-3, 1e3)

        #print(np.sum(self.noise_var.data.numpy() == 1e-2))
