import numpy as np

def regression(rows,features,seed):
    rng=np.random.default_rng(seed);x=rng.normal(size=(rows,features));w=rng.normal(size=(features,1));return x,x@w
