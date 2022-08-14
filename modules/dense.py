import numpy as np

class Dense():
    def __init__(self):
        self.n_prev = None
        self.n = None
        self.W = None
        self.B = None
        self.g = None
        self.A = None

def construct(n_prev=None, n=None, g=None, is_first=False):
    dense_obj = Dense()
    if is_first is False:
        dense_obj.n_prev = n_prev
        dense_obj.W = 2 * np.random.rand(n_prev, n) - 1
    dense_obj.n = n
    dense_obj.B = 2 * np.random.rand(1, n) - 1
    dense_obj.g = g
    return dense_obj

def forward(dense_obj=None, X=None):
    dense_obj.A = dense_obj.g(np.matmul(X, dense_obj.W) + dense_obj.B)

def print_(dense_obj=None):
    print("\nNeuron count in previous layer -> ", dense_obj.n_prev)
    print("\nNeuron count in current layer -> ", dense_obj.n)
    print("\nWeights -> \n", dense_obj.W)
    print("\nBiases -> \n", dense_obj.B)
    print("\nActivation function -> \n", dense_obj.g)
    print("\nActivations -> \n", dense_obj.A)
