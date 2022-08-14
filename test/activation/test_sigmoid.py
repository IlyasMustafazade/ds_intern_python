import sys
import os
import numpy as np
import torch.nn as nn
import timeit
import torch as tor

FILE_DEPTH = 3
sys.path.append("\\".join(os.path.abspath(__file__).split("\\")[:-FILE_DEPTH]))
import modules.activation as activation

def main():
    shape = (10000, 10000)
    X = 2 * np.random.rand(*shape) - 1

    start_relu = timeit.default_timer()
    sigmoid_X = activation.sigmoid(X)
    end_relu = timeit.default_timer()

    print("Initial -> ", X)
    print("Sigmoid -> ", sigmoid_X)
    print("Time elapsed -> ", end_relu - start_relu)


if __name__ == "__main__":
    main()
