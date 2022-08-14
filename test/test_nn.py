import unittest 
import sys
import os
import torch as tor
import numpy as np

FILE_DEPTH = 2
sys.path.append("\\".join(os.path.abspath(__file__).split("\\")[:-FILE_DEPTH]))
import modules.nn as nn
import modules.activation as activation

class TestNN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input_size, cls.input_dim = (2, 2)
        input_matrix = np.arange(4)
        cls.input_matrix = np.reshape(input_matrix, (cls.input_size, cls.input_dim))

    def test_forward(self):
        eta = 0.01
        network = nn.construct(eta=eta)
        n_neuron = 2
        nn.add_layer(nn_obj=network, n_neuron=n_neuron, g=activation.relu)
        nonrandom_weight_matrix = np.arange(TestNN.input_dim * n_neuron) 
        nonrandom_weight_matrix = np.reshape(
            nonrandom_weight_matrix, (TestNN.input_dim, n_neuron))
        network.layer_arr[0].W = nonrandom_weight_matrix
        nonrandom_bias_matrix = np.arange(3, n_neuron + 3)
        nonrandom_bias_matrix = np.reshape(
            nonrandom_bias_matrix, (n_neuron, 1))
        network.layer_arr[0].B = nonrandom_bias_matrix
        Y_h = nn.forward(nn_obj=network, X=TestNN.input_matrix)
        correct_output = np.array([
            [0, 0],
            [2, 7]
        ])
        self.assertTrue((network.layer_arr[0].B == np.array([
            [3],
            [4]
        ])).all())
        self.assertTrue((network.layer_arr[0].W == np.array([
            [0, 1],
            [2, 3]
        ])).all())
        self.assertTrue((Y_h == correct_output).all())

if __name__ == "__main__": unittest.main(verbosity=3)