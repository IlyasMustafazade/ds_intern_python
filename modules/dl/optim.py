import numpy as np


class Optimizer:
    def __call__(self, net):
        self.step(net)

    def step(self, net):
        pass


class GD(Optimizer):
    def __init__(self, eta):
        self.eta = eta

    def step(self, net):
        for i, lyr in enumerate(net.lyr_lst[1:], start=1):
            lyr.wgt = lyr.wgt - self.eta * lyr.d_wgt
            lyr.bias = lyr.bias - self.eta * lyr.d_bias
