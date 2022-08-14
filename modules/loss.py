import numpy as np

def l1(Y_h, Y):
    diff_avg = Y - Y_h
    return np.abs(diff_avg, out=diff_avg).mean()

def l2(Y_h, Y):
    diff_avg = Y - Y_h
    return np.square(diff_avg, out=diff_avg).mean()

def ce(Y_h, Y):
    return (-(Y * np.log(Y_h) + (1 - Y) * np.log(1 - Y_h))).mean()
