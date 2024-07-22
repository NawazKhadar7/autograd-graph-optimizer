import numpy as np
from .common import validate_case
from .tensor import Tensor
from .operators import numerical_gradient,mse,cross_entropy
from .graph import simplify,evaluate,node_count
FAMILIES=('scalar','broadcast','matmul','linear','mlp','softmax')
def run_case(case):
    validate_case(case)
    if case['family'] not in FAMILIES:raise ValueError('unknown autograd family')
    rng=np.random.default_rng(case['seed']);n=case['size'];f=case['family'];x=rng.normal(size=(2,n));other=rng.normal(size=(n,3));bias=rng.normal(size=(1,n))
    def objective(arr):
        t=Tensor(arr)
        if f=='scalar':loss=(t*t+2*t).mean()
        elif f=='broadcast':loss=((t+Tensor(bias,False)).power(2)).mean()
        elif f=='matmul':loss=(t@Tensor(other,False)).power(2).mean()
        elif f=='linear':loss=mse(t@Tensor(other,False),np.ones((2,3)))
        elif f=='mlp':loss=(t@Tensor(other,False)).sigmoid().mean()
        else:loss=cross_entropy(t,[0,min(1,n-1)])
        return t,loss
    t,loss=objective(x);loss.backward();numeric=numerical_gradient(lambda a:float(objective(a)[1].data),x)
    graph=('add',('mul','x',1),('add',2,3));optimized=simplify(graph)
    error=float(np.max(np.abs(t.grad-numeric)))
    return {'metrics':{'elements':x.size,'loss':float(loss.data),'max_gradient_error':error,'finite_gradients':bool(np.isfinite(t.grad).all()),'graph_nodes_before':node_count(graph),'graph_nodes_after':node_count(optimized),'graph_equivalent':evaluate(graph,{'x':3})==evaluate(optimized,{'x':3})},'output':{'gradient':t.grad.tolist(),'optimized_graph':optimized}}
