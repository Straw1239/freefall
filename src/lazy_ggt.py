from util import DirectSum, zeros_like, dnorm, ddot


class LaggedGGT:

    def __init__(params, lr=0.0001, usage=0.7, decay=0.99, pullback=):
        self.param_list = params
        self.lr = lr
        self.usage = usage
        self.decay = decay
        self.pullback = pullback
        self.lagged = zeros_like(params)


    def step():
        step = zeros_like(self.params)
        grad = DirectSum([x.grad for x in self.param_list])
        lag_comp = self.decay * ddot(grad, self.lagged)
        step = grad * (lr + (lag_comp / pow(dnorm(grad), 1.5)))
        step -= self.lagged * pullback
        self.lagged -= self.lagged * lag_comp / dnorm(grad)
        self.lagged *= decay
        for p ,s in zip(self.param_list, step):
            p -= s
        self.lagged += grad


    def zero_grad():
        for p in self.param_list:
            p.grad.detach_()
            p.grad.zero_()
    
            

        
        
    
    
