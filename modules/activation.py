import numpy as np


class ReLU:
    def function(X):
        return np.maximum(0, X)

    def derivative(X):
        return X > 0


class Sigmoid:
    def function(X):
        return 1 / (1 + np.exp(-X))

    def derivative(X):
        sigmoid_x = Sigmoid.function(X)
        return np.multiply(sigmoid_x, (1 - sigmoid_x))


class Softmax:
    def function(X):
        exp_X = np.exp(X)
        return exp_X / np.sum(exp_X, axis=1)

    def derivative(X):
        raise NotImplementedError
