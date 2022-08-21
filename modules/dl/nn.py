import sys
import os
import numpy as np


FILE_DEPTH = 3
sys.path.append(
    "\\".join(os.path.abspath(__file__).split("\\")[:-FILE_DEPTH]))
from modules.dl.layer import InputLayer
from modules.dl.activ import identity

class NN:
    def __init__(self, in_dim=None, learning_rate=None, loss_func=None):
        self.layer_lst = [InputLayer(in_dim)]
        self.learning_rate = learning_rate
        self.output = None
        self.loss_func = loss_func

    def __str__(self):
        layer_lst_str = "".join([lyr.__str__() for lyr in self.layer_lst])
        return f"\nlyr list ->\n{layer_lst_str}\
                 \nLearning rate -> {self.learning_rate}\
                 \nOutput mtx ->\n{self.output}"
    
    def add(self, lyr_type, n_neuron, activ_func):
        self.layer_lst.append(
            lyr_type(self.layer_lst[-1].out_dim, n_neuron, activ_func))
    
    def add_multi(self, attr_tpl_lst):
        for attr_tpl in attr_tpl_lst:
            lyr_type, n_neuron, activ_func = attr_tpl
            self.add(lyr_type, n_neuron, activ_func)

    def forward(self, in_mtx=None):
        last_activ = in_mtx
        for lyr in self.layer_lst:
            last_activ = lyr.forward(in_mtx=last_activ)
        self.output = last_activ

    def backward(self, label=None):
        last_lyr = self.layer_lst[-1]
        sec_last_lyr = self.layer_lst[-2]
        activ_deriv = last_lyr.activ_func(last_lyr.output, deriv=True)
        loss_deriv = self.loss_func(output=last_lyr.activ,
            actual=label, deriv=True)
        delta = np.multiply(loss_deriv, activ_deriv)
        last_lyr.set_deriv_wrt_wgt(sec_last_lyr.activ @ delta.T)
        
        for i in reversed(range(1, len(self.layer_lst) - 1)):
            left, middle, right = self.layer_lst[i - 1], self.layer_lst[i], self.layer_lst[i + 1]

            wgt_times_delta = right.wgt @ delta
            activ_deriv = middle.activ_func(middle.output, deriv=True)
            delta = np.multiply(wgt_times_delta, activ_deriv)
            middle.set_deriv_wrt_wgt(left.activ @ delta.T)
