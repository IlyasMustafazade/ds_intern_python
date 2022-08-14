import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main(): pass

# 1

def print_vec(): print(np.arange(15, 56, 1)[1:-1])

# 2

def full_arr_3_4(): return np.full((3, 4), 1.)

# 3

def arr_3_3(): return np.reshape(np.arange(10, 19), (3, 3))

# 4

def diag_matrix():

    matr = np.zeros((5, 5))

    np.fill_diagonal(matr, np.arange(1, 6))

    return matr

# 5

def null_vec():

    vec = np.zeros((10,))

    vec[5] = 11

    return vec

# 6

def to_float(lst = [1, 2, 3, 4, 5]): return np.asfarray(lst)
    
# 7

def reverse_col(arr):

    if arr.ndim != 2: raise Exception("argument must be 2d matrix")

    for i in range(np.shape(arr)[0]): arr[i] = arr[i][::-1]

    return arr

# 8

def capitalize_arr(arr): return np.char.capitalize(arr)

def lower_arr(arr): return np.char.lower(arr)

def upper_arr(arr): return np.char.upper(arr)

def swapcase_arr(arr): return np.char.swapcase(arr)

def title_arr(arr): return np.char.title(arr)

# 9

def np_dates():

    return (np.datetime64('today', 'D') - np.timedelta64(1, 'D'),
            np.datetime64('today', 'D'), np.datetime64('today', 'D')
              + np.timedelta64(1, 'D'))

# 10

def arr_append(arr = np.arange(1, 4) * 10): return np.append(arr, np.arange(4, 10) * 10)

# 11

cars = pd.read_excel('cars.xlsx').dropna()

numeric_df = cars.select_dtypes("number")

# 1

def comp_slope(x_arr, y_arr):

    n = len(x_arr)

    sum_x_arr = np.sum(x_arr)

    return (n * np.sum(x_arr * y_arr) - sum_x_arr * np.sum(y_arr)) / \
           (n * np.sum(x_arr * x_arr) - sum_x_arr ** 2)


def comp_b(x_arr, y_arr, m): return (np.sum(y_arr) - m * np.sum(x_arr)) / len(x_arr)

def least_sq(x_arr, y_arr):

    m = comp_slope(x_arr, y_arr)

    return m, comp_b(x_arr, y_arr, m)


def scatter_plt():

    x, y = cars.cty, cars.hwy

    m, b = least_sq(x, y)

    plt.scatter(x, y)

    plt.plot(x, m * x + b, color = "red")

    plt.xlabel("city")

    plt.ylabel("highway")

    plt.show()

    return "highway and city miles are linearly correlated: highway ≈ " + \
          str(round(m, 3)) + " * city + " + str(round(b, 3))

# 2

def plot_all(): sns.pairplot(cars, hue = 'drv')

def comp_r(x, y):

    n = len(x)

    x_bar, y_bar = np.sum(x) / n, np.sum(y) / n

    numerator = (x @ y - n * x_bar * y_bar)

    denominator = ((np.sum(x ** 2) - n * x_bar ** 2) * \
                  (np.sum(y ** 2) - n * y_bar ** 2)) ** 0.5

    return np.abs(numerator / denominator)


def comp_extreme_r():

    numeric_df_col = numeric_df.columns

    coeff_dict, len_numeric, start = {}, len(numeric_df_col), 0

    for i in range(len_numeric - 1):

        for j in range(start, len_numeric):

            x, y = numeric_df[numeric_df_col[i]].to_numpy(), \
                   numeric_df[numeric_df_col[j]].to_numpy()

            if x is not y:

                coeff_dict[(numeric_df_col[i], \
                      numeric_df_col[j])] = comp_r(x, y)

        start += 1

    r_val_lst, r_key_lst = list(coeff_dict.values()), list(coeff_dict.keys())

    min_key, max_key = r_key_lst[r_val_lst.index(min(r_val_lst))], \
                       r_key_lst[r_val_lst.index(max(r_val_lst))]

    return (min_key, coeff_dict[min_key]), \
           (max_key, coeff_dict[max_key])

# 3 

def plot_count(): sns.countplot(data=cars, x='class')

def comp_most_freq_type():

    class_arr = list(cars['class'].to_numpy())

    class_uniq_arr = list(set(class_arr))

    len_class_uniq_arr = len(class_uniq_arr)

    class_dict = {class_uniq_arr[i]: 0 for i in range(len_class_uniq_arr)}
    
    for i in range(len_class_uniq_arr):
        class_dict[class_uniq_arr[i]] = class_arr.count(class_uniq_arr[i])
    
    max_key = list(class_dict.keys())[class_uniq_arr.index(max(class_uniq_arr))]
    
    return (max_key, class_dict[max_key])

# 4

def plot_barchart(): sns.barplot(x='drv', y='cyl', data=cars)

def comp_most_freq_train():

    grouped, drv_dict = cars.groupby(cars.drv), {'f': [], '4': [], 'r': []}

    key_lst = list(drv_dict.keys())

    for i in key_lst:

        cyl_lst = list(grouped.get_group(i)['cyl'].to_numpy())

        drv_dict[i] = sum(cyl_lst) / len(cyl_lst)

    val_lst = list(drv_dict.values())

    max_key = key_lst[val_lst.index(max(val_lst))]

    return (max_key, drv_dict[max_key])

# 5 

def visualize_displ():

    sns.catplot(x='class', y='displ', data=cars, kind="box")

    sns.catplot(x='class', y='displ', data=cars, kind="violin")
    

if __name__ == "__main__": main()

