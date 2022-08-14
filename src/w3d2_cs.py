import numpy as np
from scipy import stats
import pandas as pd

def main(): pass

TRAIN = pd.read_csv("train.csv")

# 1

def print_head(): print(TRAIN.head(7))

# 2

def has_null(): return TRAIN.isnull().any().any()

# 3

def target_means():

    return pd.concat(objs=[TRAIN[TRAIN['target'] == 0].groupby('target').mean(),
                TRAIN[TRAIN['target'] == 1].groupby('target').mean()], axis=0)

# 4

def count_target_vals():

    return {0: int(TRAIN[TRAIN['target'] == 0].groupby('target').count()['var_0']), \
         1: int(TRAIN[TRAIN['target'] == 1].groupby('target').count()['var_0'])}  

# 5

def drop_col(col_name='ID_code'): return TRAIN.drop('ID_code', axis=1)

# 6

# a

def stat_sum(): return TRAIN.iloc[:, 1:].describe()

# b

def get_mean(var_name='var_0'): return stat_sum().loc['mean', var_name]

# 7

def filter_min_to_mean(var_name='var_0'):

    mean = stat_sum().loc['mean', var_name]

    return TRAIN[TRAIN[var_name] <= mean]

# 8

def filter_col(): return TRAIN[TRAIN['var_6'] > 5]['var_6'], \
                         TRAIN[TRAIN['var_196'] < 2]['var_196']

# 9

def filter_bound():

    col_name = TRAIN.columns[10]

    return TRAIN[(TRAIN[col_name] < 30000) &
                 (TRAIN[col_name] > 20000)][col_name]

# 10

variable_matr = []

for i in range(2, 12): variable_matr.append(list(TRAIN[TRAIN.columns[i]]))

variable_matr = np.array(variable_matr)

# a

def comp_corr(x, y):

    n = len(x)

    x_bar, y_bar = np.sum(x) / n, np.sum(y) / n

    numerator = (x @ y - n * x_bar * y_bar)

    denominator = ((np.sum(x ** 2) - n * x_bar ** 2) * \
                  (np.sum(y ** 2) - n * y_bar ** 2)) ** 0.5

    return np.abs(numerator / denominator)


def comp_corr_dict():

    coeff_dict, start = {}, 0

    col_name_lst = TRAIN.columns[2:12]

    len_col_name_lst = len(col_name_lst)

    for i in range(len_col_name_lst - 1):

        for j in range(start, len_col_name_lst):

            x, y = variable_matr[i], variable_matr[j]

            if col_name_lst[i] is not col_name_lst[j]:

                coeff_dict[(col_name_lst[i], col_name_lst[j])] = \
                             comp_corr(x, y)

        start += 1

    return coeff_dict

# b, c

def sort_corr(): 

    return sorted(comp_corr_dict().items(),
                  key=lambda x:x[1], reverse=True)

# 11

def clean_outlier():

    copy_df = TRAIN.iloc[:, 1:]

    return TRAIN.iloc[:, 1:][(np.abs(stats.zscore(copy_df)) < 3).all(axis=1)]


if __name__ == "__main__": main()


