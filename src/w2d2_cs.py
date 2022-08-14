import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main(): pass

# 1

def create_linspace(): return np.linspace(0, 1, 20, endpoint=False)

# 2

def create_matr(): return np.reshape(np.arange(1, 26), (5, 5))

def sum_cols(): 

    matr, sum_col = create_matr(), np.zeros(5)

    for i in matr.T: sum_col += i

    return np.array([sum_col]).T

# 3

def create_given_matr(): return np.reshape(np.arange(0.01, 1.01, 0.01), (10, 10))

# 4

def create_new_dt():
    
    new_dt = np.dtype([('name', 'S20'), ('surname', 'S20'), 
                      ('age', 'i1'), ('mark', 'f4')])

    return np.array(
                       [
                        ("Behram", "Abbasov", 26, 85),
                        ("Yusif", "Abdullayev", 22, 92),
                        ("Maryam", "Mecidova", 19, 88),
                        ("Vagif", "Hesenzade", 24, 79),
                       ],
                    dtype=new_dt)

# 5

def create_99_arr():

    arr = np.full((10, 10), 99)

    for i in range(1, 9): arr[i][1:9] = 0

    (arr[1][1], arr[8][8], arr[2][2:8], \
                arr[7][2:8], arr[4][4:6], arr[5][4:6]) = \
                            tuple([1 for i in range(6)])

    for i in range(3, 7): arr[i][2], arr[i][7] = 1, 1

    return arr


if __name__ == "__main__": main()

