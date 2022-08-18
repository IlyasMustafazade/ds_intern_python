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
        return 2 * (output - actual)


class LogLoss:
    def function(output=None, actual=None):
        INCREMENT = 1e-15
        return (-(np.multiply(actual, np.log(output + INCREMENT)) +\
                np.multiply((1 - actual), np.log(1 - output + INCREMENT)))).mean()

    def derivative(output=None, actual=None):
        INCREMENT = 1e-15
        return (output - actual) / (np.multiply(output, (1 - output)) + INCREMENT)

