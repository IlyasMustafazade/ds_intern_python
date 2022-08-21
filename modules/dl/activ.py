import numpy as np


def identity(X, deriv=None):
    if deriv is True: return identity_deriv(X)
    return X

def identity_deriv(X):
    return np.ones(X.shape)


def relu(X, deriv=None):
    if deriv is True: return relu_deriv(X)
    return np.maximum(0, X)

def relu_deriv(X):
    return (X > 0).astype(X.dtype)


def sigmoid(X, deriv=None):
    if deriv is True: return sigmoid_deriv(X)
    return 1 / (1 + np.exp(-X))

def sigmoid_deriv(X):
    sigmoid_x = sigmoid(X)
    return np.multiply(sigmoid_x, (1 - sigmoid_x))


def softmax(X, deriv=None):
    if deriv is True: return softmax_deriv(X)  
    exp_X = np.exp(X)
    return exp_X / np.sum(exp_X, axis=1)

def softmax_deriv(X):
    raise NotImplementedError
