import torch

from filter import kalman_update, square


#TODO make a fast approximate instead of exact calculation
def fastncdf(x, mu, sigma2):
    return 0.5 * (1 + torch.erf((x - mu) / torch.sqrt(sigma2 * 2)))


def expand_front(x, n):
    return x.expand([n] + list(x.size()))

#Can be replaced with analytical integral?
#
def mc_sign_change(grad, var, delta, dvar, n_samples=10):
    samples = torch.normal(expand_front(grad, n_samples), expand_front(torch.sqrt(var), n_samples))
    prob_positive = 1 - fastncdf(0, samples + delta, dvar)
    signs = torch.sign(samples)
    prob_changed = (samples > 0).float() - signs * prob_positive
    return prob_changed.mean(0)




class SRProp(object):
    def __init__(self, params, step=0.001, multipliers=(0.5, 1.2)):
        self.optimizers = [SRPropT(p, step, multipliers) for p in params]

    def step(self, grad_est, obs_var, delta_est, delta_var, prob_improved):
        for o, ge, ov, de, dv in zip(self.optimizers, grad_est, obs_var, delta_est, delta_var):
            o.step(ge, ov, de, dv, prob_improved)

    def zero_grad(self):
        for o in self.optimizers:
            o.zero_grad()




class SRPropT(object):
    def __init__(self, params, step, multipliers, forget=0.9):
        self.params = params
        self.steps = torch.ones_like(params) * step
        (self.n_mult, self.p_mult) = multipliers
        self.grad_mean = torch.zeros_like(params)
        self.grad_var = 100 * torch.ones_like(params)
        self.noise_var = 1e-5 * torch.ones_like(params)
        self.avg_grad_decay = 0.15 * torch.ones_like(params)
        self.prev_move_frac = torch.ones_like(params)
        self.avg_obs_noise = 1e-5*torch.ones_like(params)
        self.forget = forget
        self.prev_move_frac = torch.ones_like(params)
        self.grad_delta_cov = torch.zeros(1)
        self.grad_norm2 = torch.ones(1)
        self.delta_norm2 = torch.ones(1)

    def step(self, grad_est, obs_var,  delta_est, delta_var, prob_improved):
        
        sign_change_prob = mc_sign_change(self.grad_mean, self.grad_var, delta_est, delta_var)
        #sign_change_prob = mc_sign_change(grad_est, obs_var, delta_est, delta_var)
        #self.params.data -= self.prev_move_frac * self.steps * (1 - prob_improved)
        #self.steps *= 0.7 ** (1 - prob_improved)


        self.avg_obs_noise = self.forget * self.avg_obs_noise + (1 - self.forget) * obs_var

        noise_forget = 0.9
        self.noise_var = noise_forget * self.noise_var + (1-noise_forget) * torch.clamp(square(delta_est) - delta_var, 1e-15, 1e15)

        linear_forget = 0.95
        self.grad_delta_cov += (1 - linear_forget) * torch.dot(delta_est.view(-1),  grad_est.view(-1))
        self.grad_delta_cov *= linear_forget
        self.grad_norm2 += (1 - linear_forget) * torch.dot(self.grad_mean.view(-1), self.grad_mean.view(-1))
        self.grad_norm2 *= linear_forget
        self.delta_norm2 += (1 - linear_forget) * torch.dot(delta_est.view(-1), delta_est.view(-1))
        self.delta_norm2 *= linear_forget
        est_decay = self.grad_delta_cov / torch.sqrt(self.grad_norm2)
        #print((torch.mean(self.grad_var) * torch.mean((delta_var))))
        print(est_decay.numpy())

        est_decay.clamp_(-1, 1)

        kalman_update(self.grad_mean, self.grad_var, self.noise_var, self.avg_obs_noise, grad_est, 1 + est_decay)
        #print(self.params.detach().numpy(), self.grad_mean.numpy(), self.grad_var.numpy(), self.noise_var.numpy())
        #print(self.params.detach().numpy(), grad_est.numpy(), self.grad_mean.numpy(), torch.sqrt(self.grad_var).numpy(), self.steps.numpy(), sign_change_prob.numpy())
        #print(self.params.detach().norm().numpy())
        move_frac = (2 * fastncdf(0, self.grad_mean, self.grad_var) - 1)
        #move_frac = -torch.sign(self.grad_mean)
        self.steps *= torch.pow(self.n_mult, sign_change_prob) * torch.pow(self.p_mult, torch.abs(self.prev_move_frac) * (1 - sign_change_prob))
        #self.steps.clamp_(1e-12, 0.01)
        #self.params.data += move_frac * self.steps
        #self.params.data -= torch.sign(grad_est) * self.steps
        self.params.data -= 0.1 * grad_est

        self.prev_move_frac = move_frac

        #self.params.data += (2 * fastncdf(0, grad_est, obs_var) - 1) * self.steps

        #self.steps.clamp_(1e-6, 1)

    def zero_grad(self):
        self.params.grad.zero_()




