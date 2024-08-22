import numpy as np
from .tensor import Tensor

def mse(pred,target):return ((pred-Tensor.wrap(target)).power(2)).mean()
def cross_entropy(logits,labels):
    if logits.data.ndim!=2:raise ValueError('logits must be a matrix')
    labels=np.asarray(labels,dtype=int)
    if labels.shape!=(logits.data.shape[0],) or np.any(labels<0) or np.any(labels>=logits.data.shape[1]):raise ValueError('invalid labels')
    shifted=logits-Tensor(logits.data.max(axis=1,keepdims=True),False)
    log_probs=shifted-shifted.exp().sum(axis=1,keepdims=True).log()
    return -log_probs[np.arange(len(labels)),labels].mean()
def numerical_gradient(fn,array,epsilon=1e-5):
    values=np.asarray(array,dtype=float).copy();gradient=np.zeros_like(values)
    for index in np.ndindex(values.shape):
        old=values[index];values[index]=old+epsilon;plus=fn(values.copy());values[index]=old-epsilon;minus=fn(values.copy());values[index]=old
        gradient[index]=(plus-minus)/(2*epsilon)
    return gradient
