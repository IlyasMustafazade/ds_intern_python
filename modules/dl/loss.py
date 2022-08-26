import numpy as np


def l1(output, act, deriv=False):
    if deriv is True: return l1_deriv(output, act)
    return np.abs(act - output).mean()


def l1_deriv(output, act):
    raise NotImplementedError


def l2(output, act, deriv=False):
    if deriv is True: return l2_deriv(output, act)
    return np.square(act - output).mean()
    

def l2_deriv(output, act):
    return (2 * (output - act)) / output.size


def logloss(output, act, deriv=False):
    if deriv is True: return logloss_deriv(output, act)
    INCR = 1e-16
    return (-(act * np.log(output + INCR) +\
             (1 - act) * np.log(1 - output + INCR))).mean()


def logloss_deriv(output, act):
    INCR = 1e-16
    return (output - act) / (output * (1 - output) * output.size + INCR)
