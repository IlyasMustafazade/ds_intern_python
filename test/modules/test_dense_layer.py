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
        cls.n_neuron = 3
        cls.EPSILON = 1e-4
        cls.layer = None
    
    def setUp(self):
        TestDenseLayer.layer = dense_layer.DenseLayer(input_dim=TestDenseLayer.input_dim,
                                       output_dim=TestDenseLayer.n_neuron,
                                       activation_func=activation.ReLU) 

    def test_init(self):            
        self.assertEqual(TestDenseLayer.layer.output_dim, TestDenseLayer.n_neuron)
        self.assertEqual(TestDenseLayer.layer.input_dim, TestDenseLayer.input_dim)
        self.assertIs(TestDenseLayer.layer.activation_func, activation.ReLU)
        self.assertIsNone(TestDenseLayer.layer.net_output)
        self.assertIsNone(TestDenseLayer.layer.activation_matrix)
        self.assertIsNone(TestDenseLayer.layer.derivative_wrt_weight)
        self.assertIsNone(TestDenseLayer.layer.derivative_wrt_bias)

    
    def test_forward_pass(self):
        n_row_weight, n_col_weight = TestDenseLayer.layer.weight_matrix.shape
        nonrandom_weight_matrix = np.arange(n_row_weight * n_col_weight)
        nonrandom_weight_matrix = np.reshape(nonrandom_weight_matrix, (n_row_weight, n_col_weight))
        TestDenseLayer.layer.weight_matrix = nonrandom_weight_matrix
        self.assertTrue((TestDenseLayer.layer.weight_matrix == np.array([
            [0, 1, 2],
            [3, 4, 5]
        ])).all())

        n_row_bias, n_col_bias = TestDenseLayer.layer.bias_matrix.shape
        nonrandom_bias_matrix = np.arange(10, 10 + n_row_bias * n_col_bias)
        nonrandom_bias_matrix = np.reshape(nonrandom_bias_matrix, (n_row_bias, n_col_bias))
        nonrandom_bias_matrix = -nonrandom_bias_matrix
        TestDenseLayer.layer.bias_matrix = nonrandom_bias_matrix
        self.assertTrue((TestDenseLayer.layer.bias_matrix == np.array([
            [[-10, -11, -12]]
        ])).all())

        TestDenseLayer.layer.forward_pass(input_matrix=TestDenseLayer.input_matrix)
        self.assertTrue((TestDenseLayer.layer.net_output == np.array([
            [-7, -7, -7],
            [-1, 3, 7],
            [5, 13, 21]
        ])).all())

        self.assertTrue((TestDenseLayer.layer.activation_matrix == np.array([
            [0, 0, 0],
            [0, 3, 7],
            [5, 13, 21]
        ])).all())
        print(TestDenseLayer.layer)
    
    def test_increment_weight(self):
        old_weight = TestDenseLayer.layer.weight_matrix
        TestDenseLayer.layer.increment_weight()
        new_weight = TestDenseLayer.layer.weight_matrix
        np.testing.assert_allclose(
            new_weight, old_weight + TestDenseLayer.EPSILON, rtol=1e-8)

    def test_decrement_weight(self):
        old_weight = TestDenseLayer.layer.weight_matrix
        TestDenseLayer.layer.decrement_weight()
        new_weight = TestDenseLayer.layer.weight_matrix
        np.testing.assert_allclose(
            new_weight, old_weight - TestDenseLayer.EPSILON, rtol=1e-8)
    
    def test_increment_bias(self):
        old_bias = TestDenseLayer.layer.bias_matrix
        TestDenseLayer.layer.increment_bias()
        new_bias = TestDenseLayer.layer.bias_matrix
        np.testing.assert_allclose(
            new_bias, old_bias + TestDenseLayer.EPSILON, rtol=1e-8)
    
    def test_decrement_bias(self):
        old_bias = TestDenseLayer.layer.bias_matrix
        TestDenseLayer.layer.decrement_bias()
        new_bias = TestDenseLayer.layer.bias_matrix
        np.testing.assert_allclose(
            new_bias, old_bias - TestDenseLayer.EPSILON, rtol=1e-8)
        
    def test_increment_decrement_weight(self):
        old_weight = TestDenseLayer.layer.weight_matrix
        TestDenseLayer.layer.increment_weight()
        TestDenseLayer.layer.decrement_weight()
        new_weight = TestDenseLayer.layer.weight_matrix
        np.testing.assert_allclose(
            new_weight, old_weight, rtol=1e-8)

    def test_increment_decrement_bias(self):
        old_bias = TestDenseLayer.layer.bias_matrix
        TestDenseLayer.layer.increment_bias()
        TestDenseLayer.layer.decrement_bias()
        new_bias = TestDenseLayer.layer.bias_matrix
        np.testing.assert_allclose(
            new_bias, old_bias, rtol=1e-8)


if __name__ == "__main__": unittest.main(verbosity=3)

