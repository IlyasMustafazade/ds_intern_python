import sys
import os

FILE_DEPTH = 3
sys.path.append("\\".join(os.path.abspath(__file__).split("\\")[:-FILE_DEPTH]))
import modules.nn as nn
import modules.dense as dense
import modules.activation as activation

def main():
    network = nn.construct(eta=0.01)
    nn.print_(nn_obj=network)
    nn.add_layer(nn_obj=network, n_neuron=6, g=activation.sigmoid)
    nn.print_(nn_obj=network)
    dense.print_(dense_obj=network.layer_arr[0])
    nn.add_layer(nn_obj=network, n_neuron=4, g=activation.relu)
    nn.print_(nn_obj=network)
    dense.print_(dense_obj=network.layer_arr[1])

if __name__ == "__main__":
    main()
