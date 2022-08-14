import sys
import os
import numpy as np

FILE_DEPTH = 3
sys.path.append("\\".join(os.path.abspath(__file__).split("\\")[:-FILE_DEPTH]))
import modules.dense as dense
import modules.activation as activation

def main():
    X = np.reshape(np.arange(1, 7), (3, 2))
    n_prev1, n1 = 2, 4
    is_first = False
    layer = dense.construct(n=n1, n_prev=n_prev1, g=activation.relu, is_first=is_first)
    dense.forward(dense_obj=layer, X=X)
    n_prev2, n2 = n1, 5
    layer2 = dense.construct(n=n2, n_prev=n_prev2, g=activation.sigmoid, is_first=is_first)
    dense.forward(dense_obj=layer2, X=layer.A)
    print("X -> \n", X)
    dense.print_(dense_obj=layer)
    dense.print_(dense_obj=layer2)


if __name__ == "__main__":
    main()
