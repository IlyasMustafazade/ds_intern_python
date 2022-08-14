import sys
import os
import numpy as np

FILE_DEPTH = 3
sys.path.append("\\".join(os.path.abspath(__file__).split("\\")[:-FILE_DEPTH]))
import modules.nn as nn

def main():
    network = nn.construct(eta=0.01)
    print(network)
    print(network.eta)
    print(network.layer_arr)
    nn.print_(nn_obj=network)

if __name__ == "__main__":
    main()
