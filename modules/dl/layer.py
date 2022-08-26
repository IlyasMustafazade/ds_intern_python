import numpy as np, sys, os, logging

FILE_DEPTH = 3
sys.path.append(
    "\\".join(os.path.abspath(__file__).split("\\")[:-FILE_DEPTH]))
from modules.dl.activ import identity


class Layer:
    pass


class Input(Layer):
    def __init__(self, out_dim):
        self.out_dim = out_dim
        self.activ_fn = identity
        self.output = None
        self.activ = None

    def forward(self, in_mtx, save=True):
        if save is True:
            self.output = in_mtx
            self.activ = self.output
        return in_mtx


class Dense(Layer):
    def __init__(self, in_dim, out_dim, activ_fn):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.wgt = self.init_wgt()
        self.bias = self.init_bias()
        self.output = None
        self.activ_fn = activ_fn
        self.activ = None
        self.d_wgt = np.zeros(self.wgt.shape)
        self.d_bias = np.zeros(self.bias.shape)

    def init_wgt(self):
        rand = np.random.rand(self.in_dim, self.out_dim)
        return 2 * rand - 1

    def init_bias(self):
        rand = np.random.rand(self.out_dim, 1)
        return 2 * rand - 1

    def forward(self, in_mtx, save=True):
        net_output = self.wgt.T @ in_mtx + self.bias
        activ = self.activ_fn(net_output)
        if save is True:
            self.output = net_output
            self.activ = activ
        return activ

    def set_d_wgt(self, X):
        if X.shape != self.d_wgt.shape:
            raise ValueError(
                "Cost derivative w.r.t weight must be same shape as weight")
        self.d_wgt = X

    def set_d_bias(self, X):
        if X.shape != self.d_bias.shape:
            raise ValueError(
                "Cost derivative w.r.t bias must be same shape as bias")
        self.d_bias = X
