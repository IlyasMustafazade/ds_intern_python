import sys
import os
import numpy as np

FILE_DEPTH = 3
sys.path.append("\\".join(os.path.abspath(__file__).split("\\")[:-FILE_DEPTH]))
import modules.dense as dense

class NN:
    def __init__(self):
        self.eta = None
        self.layer_arr = np.empty(0)
        self.Y_h = None

def construct(eta=None):
    network = NN()
    network.eta = eta
    return network

def add_layer(nn_obj=None, n_neuron=None, g=None):
    layer_arr = nn_obj.layer_arr
    is_first = True if len(layer_arr) < 1 else False
    if is_first is True:
        layer = dense.construct(n=n_neuron, g=g, is_first=is_first)
    else: 
        layer = dense.construct(n=n_neuron, n_prev=layer_arr[-1].n, g=g, is_first=is_first)
    nn_obj.layer_arr = np.append(layer_arr, [layer])

def print_(nn_obj=None):
    print("\nLearning rate -> ", nn_obj.eta)
    print("\nLayer array -> \n", nn_obj.layer_arr)

def forward(nn_obj=None, X=None):
    x_n = X.shape[1]
    nn_obj.layer_arr[0].n_prev = x_n
    nn_obj.layer_arr[0].W = 2 * np.random.rand(x_n, nn_obj.layer_arr[0].n) - 1

    A = X
    for layer in nn_obj.layer_arr:
        dense.forward(dense_obj=layer, X=A)
        A = layer.A
    return A
