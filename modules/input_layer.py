import numpy as np


class InputLayer:
    def __init__(self,
                 output_dim=None):
        self.output_dim = output_dim
        self.activation_matrix = None
    
    def __str__(self):
        return f"\nNeuron count in current layer -> {self.output_dim}\
                \nActivations (same as input) -> \n{self.activation_matrix}"
    
    def forward_pass(self, input_matrix=None):
        self.activation_matrix = input_matrix
