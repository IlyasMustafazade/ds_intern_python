import numpy as np


class L1:
    def function(output=None, actual=None):
        diff_avg = actual - output
        return np.abs(diff_avg, out=diff_avg).mean()
    
    def derivative(output=None, actual=None):
        raise NotImplementedError


class L2:
    def function(output=None, actual=None):
        diff_avg = actual - output
        return np.square(diff_avg, out=diff_avg).mean()
    
    def derivative(output=None, actual=None):
        raise NotImplementedError


class LogLoss:
    def function(output=None, actual=None):
        return (-(actual * np.log(output + 1e-15) + (1 - actual) * np.log(1 - output + 1e-15))).mean()

    def derivative(output=None, actual=None):
        raise NotImplementedError
