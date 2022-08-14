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
    shape = (5, 5)
    Y_hat = np.random.rand(*shape)
    Y = np.random.rand(*shape)
    print("allocated memory")

    start_l2 = timeit.default_timer()
    e_l2 = loss.l2(Y_hat, Y)
    end_l2 = timeit.default_timer()

    Y_hat_tensor = tor.from_numpy(Y_hat)
    Y_tensor = tor.from_numpy(Y)
    loss_torch = nn.MSELoss()

    start_torch = timeit.default_timer()
    e_torch = loss_torch(Y_hat_tensor, Y_tensor)
    end_torch = timeit.default_timer()

    print("l2 -> \n", e_l2)
    print("time l2 -> ", end_l2 - start_l2)

    print("loss torch -> \n", e_torch)
    print("time loss torch -> ", end_torch - start_torch)


if __name__ == "__main__":
    main()
