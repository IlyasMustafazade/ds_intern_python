import numpy as np


def identity(x, deriv=False):
    if deriv is True: return identity_deriv(x)
    return x


def identity_deriv(x):
    return np.ones(x.shape)


def relu(x, deriv=False):
    if deriv is True: return relu_deriv(x)
    return np.maximum(0, x)


def relu_deriv(x):
    return (x > 0).astype(x.dtype)


def sigmoid(x, deriv=False):
    if deriv is True: return sigmoid_deriv(x)
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    sig_x = sigmoid(x)
    return np.multiply(sig_x, (1 - sig_x))


def softmax(x, deriv=False):
    if deriv is True: return softmax_deriv(x)  
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1)


def softmax_deriv(x):
    raise NotImplementedError
