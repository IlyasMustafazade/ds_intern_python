import sys
import os
import numpy as np
import torch as tor

FILE_DEPTH = 3
sys.path.append("\\".join(os.path.abspath(__file__).split("\\")[:-FILE_DEPTH]))
import modules.nn as nn
import modules.dense as dense
import modules.activation as activation

def main():
    shape = (3, 4)
    network = nn.construct(eta=0.01)
    nn.add_layer(nn_obj=network, n_neuron=6, g=activation.sigmoid)
    nn.add_layer(nn_obj=network, n_neuron=4, g=activation.relu)

    X = np.random.rand(*shape)
    X_tensor = tor.from_numpy(X)

    Y_h = nn.forward(nn_obj=network, X=X)
    print(Y_h)

if __name__ == "__main__":
    main()
