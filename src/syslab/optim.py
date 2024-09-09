import numpy as np
class SGD:
    def __init__(self,parameters,lr=.01):
        if lr<=0:raise ValueError('positive learning rate required')
        self.parameters=list(parameters);self.lr=lr
    def step(self):
        for p in self.parameters:
            if not np.all(np.isfinite(p.grad)):raise ValueError('nonfinite gradient')
            p.data-=self.lr*p.grad
    def zero_grad(self):
        for p in self.parameters:p.grad.fill(0)
