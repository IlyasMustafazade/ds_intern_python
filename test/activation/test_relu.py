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
    X = np.random.rand(*shape)

    start_relu = timeit.default_timer()
    relu_X = activation.relu(X)
    end_relu = timeit.default_timer()

    X_tensor = tor.from_numpy(X)
    relu_obj = nn.ReLU(inplace=True)

    start_torch = timeit.default_timer()
    relu_torch = relu_obj(X_tensor)
    end_torch = timeit.default_timer()

    print("Time ReLU -> ", end_relu - start_relu)
    print("Time Torch -> ", end_torch - start_torch)


if __name__ == "__main__":
    main()
