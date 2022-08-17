import sys
import os
import numpy as np

FILE_DEPTH = 2
sys.path.append("\\".join(os.path.abspath(__file__).split("\\")[:-FILE_DEPTH]))
from modules.input_layer import InputLayer

class NeuralNetwork:
    def __init__(self,
                 layer_lst=None, 
                 learning_rate=None):
        self.layer_lst = layer_lst
        self._check_layer_lst()
        self.learning_rate = learning_rate
        self.output_matrix = None
    
    def __str__(self):
        return f"\nLayer list -> \n {self.layer_lst}\
                 \nLearning rate -> {self.learning_rate}\
                 \nOutput matrix ->\n{self.output_matrix}"
            
    def _check_layer_lst(self):
        self._check_input_first_layer()
        self._check_unique_input()
    
    def _check_input_first_layer(self):
        first_layer = self.layer_lst[0]
        if not isinstance(first_layer, InputLayer):
            raise ValueError("First layer must be InputLayer") 
    
    def _check_unique_input(self):
        n_input = 0
        for layer in self.layer_lst:
            if isinstance(layer, InputLayer):
                n_input += 1
            if n_input >= 2:
                raise ValueError("InputLayer must be unique") 
    
    def forward(self, input_matrix=None):
        last_activation_matrix = input_matrix
        for layer in self.layer_lst:
            layer.forward_pass(input_matrix=last_activation_matrix)
            last_activation_matrix = layer.activation_matrix
        self.output_matrix = last_activation_matrix

    def backward(self, label_matrix=None):
        last_layer = self.layer_lst[-1]
        delta = np.multiply(last_layer.activation_matrix - label_matrix,
            last_layer.activation_func.derivative(last_layer.net_output))

        for i in reversed(range(len(self.layer_lst) - 1)):
            middle, right = self.layer_lst[i], self.layer_lst[i + 1]
            right.derivative_wrt_weight = np.matmul(middle.activation_matrix, delta.T)
            right.derivative_wrt_bias = delta

            delta = np.multiply(np.matmul(right.weight_matrix, delta),
                middle.activation_func.derivative(middle.net_output))

