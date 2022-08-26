import numpy as np


def extract_X_y_df(df=None):
    return df.iloc[:, :-1], df.iloc[:, -1]


def train_test_split(ftr, lbl, test_frac=None):
    in_size = len(ftr)
    all_idx = np.arange(in_size)
    test_size = round(in_size * test_frac)
    test_idx = np.random.choice(all_idx, size=test_size, replace=False)
    train_idx = all_idx[np.setdiff1d(all_idx, test_idx)]
    return (ftr[train_idx], ftr[test_idx], lbl[train_idx], lbl[test_idx])

