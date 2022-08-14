import sys
import os
import numpy as np
import sklearn.metrics as metrics
import timeit

FILE_DEPTH = 3
sys.path.append("\\".join(os.path.abspath(__file__).split("\\")[:-FILE_DEPTH]))
import modules.loss as loss

def main():
    m = 5

    Y_h = np.random.rand(m)
    print("Y_h -> ", Y_h)

    Y = np.random.randint(2, size=m)
    print("Y -> ", Y)

    ce_loss = loss.ce(Y_h, Y)
    print("Cross entropy loss -> ", ce_loss)

    print("metrics.log_loss -> ", metrics.log_loss(Y, Y_h))


if __name__ == "__main__":
    main()
