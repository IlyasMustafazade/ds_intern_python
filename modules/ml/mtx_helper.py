import numpy as np
import pandas as pd

def extract_X_y_df(df=None):
    column_name_arr = df.columns
    column_name_arr_without_last = df.columns[:-1]
    column_name_tpl_without_last = tuple(column_name_arr_without_last)
    last_column_name = column_name_arr[-1]
    X = df[[*column_name_tpl_without_last]]
    y = df[last_column_name]
    return X, y


