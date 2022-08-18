import sys
import os
import numpy as np


class NeuralNetwork:
    def __init__(self,
                 layer_lst=None, 
                 learning_rate=None,
                 loss_func=None):
        self.layer_lst = layer_lst
        self.learning_rate = learning_rate
        self.output_matrix = None
        self.loss_func = loss_func
    
    def __str__(self):
        return f"\nLayer list -> \n {self.layer_lst}\
                 \nLearning rate -> {self.learning_rate}\
                 \nOutput matrix ->\n{self.output_matrix}"
    
    def forward(self, input_matrix=None):
        last_activation_matrix = input_matrix
        for layer in self.layer_lst:
            layer.forward_pass(input_matrix=last_activation_matrix)
            last_activation_matrix = layer.activation_matrix
        self.output_matrix = last_activation_matrix

    def backward(self, label_matrix=None):
        last_layer = self.layer_lst[-1]
        activation_derivative = last_layer.activation_func.derivative(last_layer.net_output)
        loss_derivative = self.loss_func.derivative(output=last_layer.activation_matrix,
                                               actual=label_matrix)
        delta = np.multiply(loss_derivative, activation_derivative)

        for i in reversed(range(-1, len(self.layer_lst) - 1)):
            middle, right = self.layer_lst[i], self.layer_lst[i + 1]
            right.derivative_wrt_weight = np.matmul(middle.activation_matrix.T, delta)
            right.derivative_wrt_bias = delta


            weight_times_delta = np.matmul(right.weight_matrix, delta.T)
            activation_derivative = middle.activation_func.derivative(middle.net_output)
            delta = np.multiply(weight_times_delta.T, activation_derivative)

