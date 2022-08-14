import sys
import os

FILE_DEPTH = 3
sys.path.append("\\".join(os.path.abspath(__file__).split("\\")[:-FILE_DEPTH]))
import modules.dense as dense
import modules.activation as activation

def main():
    is_first = True
    n_prev, n = 3, 4
    layer = dense.construct(n=n, n_prev=n_prev, g=activation.sigmoid, is_first=is_first)
    dense.print_(dense_obj=layer)


if __name__ == "__main__":
    main()

