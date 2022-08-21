import numpy as np


def l1(output=None, actual=None, deriv=False):
    if deriv is True: return l1_deriv(output=output, actual=actual)
    return np.abs(actual - output).mean()
    
def l1_deriv(output=None, actual=None):
    raise NotImplementedError


def l2(output=None, actual=None, deriv=False):
    if deriv is True: return l2_deriv(output=output, actual=actual)
    return np.square(actual - output).mean()
    
def l2_deriv(output=None, actual=None):
    return (2 * (output - actual)) / output.size


def logloss(output=None, actual=None, deriv=False):
    if deriv is True: return logloss_deriv(output=output, actual=actual)
    INCR = 1e-15
    return (-(np.multiply(actual, np.log(output + INCR)) +\
            np.multiply((1 - actual), np.log(1 - output + INCR)))).mean()

def logloss_deriv(output=None, actual=None):
    INCR = 1e-15
    return ((output - actual) /
        (np.multiply(output, (1 - output)) + INCR)) / output.size

