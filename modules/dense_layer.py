import numpy as np


class DenseLayer:
    def __init__(self,
                 input_dim=None,
                 output_dim=None,
                 activation_func=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_matrix = self.random_init_weight()
        self.bias_matrix = self.random_init_bias()
        self.activation_func = activation_func
        self.activation_matrix = None
    
    def __str__(self):
        return f"\nNeuron count in previous layer -> {self.input_dim}\
                \nNeuron count in current layer -> {self.output_dim}\
                \nWeights -> \n{self.weight_matrix}\
                \nBiases -> \n{self.bias_matrix}\
                \nActivation function -> \n{self.activation_func}\
                \nActivations -> \n{self.activation_matrix}"
    
    def random_init_weight(self):
        weight_shape = (self.input_dim, self.output_dim)
        random_matrix = np.random.rand(*weight_shape)
        rearranged_random_matrix = 2 * random_matrix - 1
        return rearranged_random_matrix
    
    def random_init_bias(self):
        bias_shape = (1, self.output_dim)
        random_matrix = np.random.rand(*bias_shape)
        rearranged_random_matrix = 2 * random_matrix - 1
        return rearranged_random_matrix
    
    def forward_pass(self, input_matrix=None):
        product = np.matmul(input_matrix, self.weight_matrix)
        product_with_bias = product + self.bias_matrix
        self.activation_matrix = self.activation_func(product_with_bias)