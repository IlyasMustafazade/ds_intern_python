import sys
import os
import numpy as np
import torch as tor
import torch.nn as nn
import timeit

FILE_DEPTH = 3
sys.path.append("\\".join(os.path.abspath(__file__).split("\\")[:-FILE_DEPTH]))
import modules.loss as loss

def main():
    shape = (2, 6)
    Y_hat = np.random.rand(*shape)
    Y = np.random.rand(*shape)
    print("Allocated memory")

    start_l1 = timeit.default_timer()
    e_l1 = loss.l1(Y_hat, Y)
    end_l1 = timeit.default_timer()

    Y_hat_tensor = tor.from_numpy(Y_hat)
    Y_tensor = tor.from_numpy(Y)
    loss_torch = nn.L1Loss()

    start_torch = timeit.default_timer()
    e_torch = loss_torch(Y_hat_tensor, Y_tensor)
    end_torch = timeit.default_timer()

    print("l1 -> \n", e_l1)
    print("time l1 -> ", end_l1 - start_l1)

    print("loss torch -> \n", e_torch)
    print("time loss torch -> ", end_torch - start_torch)


if __name__ == "__main__":
    main()
