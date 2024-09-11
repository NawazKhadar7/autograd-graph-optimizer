import numpy as np
from .tensor import Tensor
class Linear:
    def __init__(self,inputs,outputs,seed=0):
        rng=np.random.default_rng(seed);self.weight=Tensor(rng.normal(0,1/np.sqrt(inputs),(inputs,outputs)));self.bias=Tensor(np.zeros((1,outputs)))
    def __call__(self,x):return x@self.weight+self.bias
    def parameters(self):return [self.weight,self.bias]
class MLP:
    def __init__(self,inputs,hidden,outputs,seed=0):self.a=Linear(inputs,hidden,seed);self.b=Linear(hidden,outputs,seed+1)
    def __call__(self,x):return self.b(self.a(x).relu())
    def parameters(self):return self.a.parameters()+self.b.parameters()
