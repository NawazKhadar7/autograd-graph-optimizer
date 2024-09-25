import numpy as np

def reduce_like(gradient,shape):
    while gradient.ndim>len(shape):gradient=gradient.sum(axis=0)
    for axis,size in enumerate(shape):
        if size==1 and gradient.shape[axis]!=1:gradient=gradient.sum(axis=axis,keepdims=True)
    return gradient.reshape(shape)
class Tensor:
    def __init__(self,data,requires_grad=True,parents=(),backward=None):
        self.data=np.array(data,dtype=float)
        if not np.all(np.isfinite(self.data)):raise ValueError('tensor must be finite')
        self.requires_grad=requires_grad;self.grad=np.zeros_like(self.data);self.parents=tuple(parents);self._backward=backward or (lambda:None)
    @staticmethod
    def wrap(value):return value if isinstance(value,Tensor) else Tensor(value,False)
    def add_grad(self,value):
        if self.requires_grad:self.grad+=reduce_like(np.asarray(value),self.data.shape)
    def __add__(self,other):
        b=self.wrap(other);out=Tensor(self.data+b.data,self.requires_grad or b.requires_grad,(self,b))
        def backward():self.add_grad(out.grad);b.add_grad(out.grad)
        out._backward=backward;return out
    __radd__=__add__
    def __neg__(self):return self*-1
    def __sub__(self,b):return self+-self.wrap(b)
    def __rsub__(self,b):return self.wrap(b)+-self
    def __mul__(self,other):
        b=self.wrap(other);out=Tensor(self.data*b.data,self.requires_grad or b.requires_grad,(self,b))
        def backward():self.add_grad(out.grad*b.data);b.add_grad(out.grad*self.data)
        out._backward=backward;return out
    __rmul__=__mul__
    def __truediv__(self,b):return self*self.wrap(b).power(-1)
    def power(self,n):
        out=Tensor(self.data**n,self.requires_grad,(self,));out._backward=lambda:self.add_grad(np.zeros_like(self.data) if n==0 else out.grad*n*self.data**(n-1));return out
    def __matmul__(self,other):
        b=self.wrap(other)
        if self.data.ndim!=2 or b.data.ndim!=2:raise ValueError('reference matmul requires matrices')
        out=Tensor(self.data@b.data,self.requires_grad or b.requires_grad,(self,b))
        def backward():self.add_grad(out.grad@b.data.T);b.add_grad(self.data.T@out.grad)
        out._backward=backward;return out
    def sum(self,axis=None,keepdims=False):
        out=Tensor(self.data.sum(axis=axis,keepdims=keepdims),self.requires_grad,(self,))
        def backward():
            grad=out.grad
            if axis is not None and not keepdims:grad=np.expand_dims(grad,axis)
            self.add_grad(np.broadcast_to(grad,self.data.shape))
        out._backward=backward;return out
    def mean(self):return self.sum()/self.data.size
    def exp(self):
        out=Tensor(np.exp(self.data),self.requires_grad,(self,));out._backward=lambda:self.add_grad(out.grad*out.data);return out
    def log(self):
        if np.any(self.data<=0):raise ValueError('log domain must be positive')
        out=Tensor(np.log(self.data),self.requires_grad,(self,));out._backward=lambda:self.add_grad(out.grad/self.data);return out
    def relu(self):
        out=Tensor(np.maximum(self.data,0),self.requires_grad,(self,));out._backward=lambda:self.add_grad(out.grad*(self.data>0));return out
    def sigmoid(self):
        x=self.data;positive=x>=0;y=np.empty_like(x);y[positive]=1/(1+np.exp(-x[positive]));z=np.exp(x[~positive]);y[~positive]=z/(1+z)
        out=Tensor(y,self.requires_grad,(self,));out._backward=lambda:self.add_grad(out.grad*y*(1-y));return out
    def __getitem__(self,index):
        out=Tensor(self.data[index],self.requires_grad,(self,))
        def backward():
            g=np.zeros_like(self.data);np.add.at(g,index,out.grad);self.add_grad(g)
        out._backward=backward;return out
    def backward(self):
        if self.data.size!=1:raise ValueError('backward requires a scalar loss')
        order=[];seen=set()
        def visit(node):
            if id(node) in seen:return
            seen.add(id(node))
            for parent in node.parents:visit(parent)
            order.append(node)
        visit(self)
        for node in order:node.grad=np.zeros_like(node.data)
        self.grad=np.ones_like(self.data)
        for node in reversed(order):node._backward()
