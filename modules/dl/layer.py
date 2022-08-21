import numpy as np, sys, os

FILE_DEPTH = 3
sys.path.append("\\".join(os.path.abspath(__file__).split("\\")[:-FILE_DEPTH]))
from modules.dl.activ import identity


class Layer:
    pass


class InputLayer(Layer):
    def __init__(self, out_dim):
        self.out_dim = out_dim
        self.activ_func = identity
        self.output = None
        self.activ = None

    def forward(self, in_mtx):
        self.output = in_mtx
        self.activ = self.output
        return in_mtx


class DenseLayer(Layer):
    def __init__(self, in_dim, out_dim, activ_func):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.wgt = self.init_wgt()
        self.bias = self.init_bias()
        self.output = None
        self.activ_func = activ_func
        self.activ = None
        self.deriv_wrt_wgt = np.zeros(self.wgt.shape)
        self.deriv_wrt_bias = np.zeros(self.bias.shape)

    def __str__(self):
        return f"\nNeuron count in previous layer -> {self.in_dim}\
                 \nNeuron count in current layer -> {self.out_dim}\
                 \nwgts -> \n{self.wgt}\
                 \nBiases -> \n{self.bias}\
                 \nactiv func -> \n{self.activ_func}\
                 \nNet output -> \n{self.output}\
                 \nactivs -> \n{self.activ}\
                 \nPartial derivative w.r.t wgt mtx -> \n\
                    {self.deriv_wrt_wgt}\
                 \nPartial derivative w.r.t bias mtx -> \n\
                    {self.deriv_wrt_bias}"

    def init_wgt(self):
        wgt_shape = (self.in_dim, self.out_dim)
        rand = np.random.rand(*wgt_shape)
        rand = 2 * rand - 1
        return rand

    def init_bias(self):
        bias_shape = (self.out_dim, 1)
        rand = np.random.rand(*bias_shape)
        rand = 2 * rand - 1
        return rand

    def forward(self, in_mtx):
        prod = self.wgt.T @ in_mtx
        prod_with_bias = prod + self.bias
        self.output = prod_with_bias
        self.activ = self.activ_func(self.output)
        return self.activ
    
    def set_deriv_wrt_wgt(self, X):
        if X.shape != self.deriv_wrt_wgt.shape:
            raise ValueError(
                "Cost derivative w.r.t weight must be same shape as weight")
        self.deriv_wrt_wgt = X
    
    def set_deriv_wrt_bias(self, X):
        if X.shape != self.deriv_wrt_bias.shape:
            raise ValueError(
                "Cost derivative w.r.t bias must be same shape as bias")
        self.deriv_wrt_bias = X
