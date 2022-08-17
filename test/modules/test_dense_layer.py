import unittest
import numpy as np
import sys
import os


FILE_DEPTH = 3
sys.path.append("\\".join(os.path.abspath(__file__).split("\\")[:-FILE_DEPTH]))
import modules.dense_layer as dense_layer
import modules.activation as activation


class TestDenseLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input_size, cls.input_dim = (3, 2)
        input_matrix = np.arange(cls.input_size * cls.input_dim)
        cls.input_matrix = np.reshape(input_matrix, (cls.input_size, cls.input_dim))


    def test_init(self):
        n_neuron = 3
        layer = dense_layer.DenseLayer(input_dim=TestDenseLayer.input_dim,
                                       output_dim=n_neuron,
                                       activation_func=activation.ReLU)
                                    
        self.assertEqual(layer.output_dim, n_neuron)
        self.assertEqual(layer.input_dim, TestDenseLayer.input_dim)
        self.assertIs(layer.activation_func, activation.ReLU)
        self.assertIsNone(layer.net_output)
        self.assertIsNone(layer.activation_matrix)
        self.assertIsNone(layer.derivative_wrt_weight)
        self.assertIsNone(layer.derivative_wrt_bias)

    
    def test_forward_pass(self):
        n_neuron = 3
        layer = dense_layer.DenseLayer(input_dim=TestDenseLayer.input_dim,
                                       output_dim=n_neuron,
                                       activation_func=activation.ReLU)
        n_row_weight, n_col_weight = layer.weight_matrix.shape
        nonrandom_weight_matrix = np.arange(n_row_weight * n_col_weight)
        nonrandom_weight_matrix = np.reshape(nonrandom_weight_matrix, (n_row_weight, n_col_weight))
        layer.weight_matrix = nonrandom_weight_matrix
        self.assertTrue((layer.weight_matrix == np.array([
            [0, 1, 2],
            [3, 4, 5]
        ])).all())

        n_row_bias, n_col_bias = layer.bias_matrix.shape
        nonrandom_bias_matrix = np.arange(10, 10 + n_row_bias * n_col_bias)
        nonrandom_bias_matrix = np.reshape(nonrandom_bias_matrix, (n_row_bias, n_col_bias))
        nonrandom_bias_matrix = -nonrandom_bias_matrix
        layer.bias_matrix = nonrandom_bias_matrix
        self.assertTrue((layer.bias_matrix == np.array([
            [[-10, -11, -12]]
        ])).all())

        layer.forward_pass(input_matrix=TestDenseLayer.input_matrix)
        self.assertTrue((layer.net_output == np.array([
            [-7, -7, -7],
            [-1, 3, 7],
            [5, 13, 21]
        ])).all())

        self.assertTrue((layer.activation_matrix == np.array([
            [0, 0, 0],
            [0, 3, 7],
            [5, 13, 21]
        ])).all())
        print(layer)


if __name__ == "__main__": unittest.main(verbosity=3)
