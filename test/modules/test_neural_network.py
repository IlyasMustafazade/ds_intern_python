import unittest
import numpy as np
import sys
import os


FILE_DEPTH = 3
sys.path.append("\\".join(os.path.abspath(__file__).split("\\")[:-FILE_DEPTH]))
import modules.neural_network as neural_network
import modules.activation as activation
from modules.loss import LogLoss
from modules.dense_layer import DenseLayer
from modules.differentiation import numeric_derivative


class TestNeuralNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input_size, cls.input_dim = (3, 2)
        input_matrix = np.arange(cls.input_size * cls.input_dim)
        cls.input_matrix = np.reshape(input_matrix, (cls.input_size, cls.input_dim))

        n_neuron2 = 3
        cls.layer2 = DenseLayer(input_dim=TestNeuralNetwork.input_dim,
                                output_dim=n_neuron2,
                                activation_func=activation.ReLU)

        n_neuron3 = 2
        cls.layer3 = DenseLayer(input_dim=cls.layer2.output_dim,
                                output_dim=n_neuron3,
                                activation_func=activation.ReLU)


    def test__init__(self):
        learning_rate = 0.1 
        network = neural_network.NeuralNetwork(
            layer_lst=[TestNeuralNetwork.layer2,  
                       TestNeuralNetwork.layer3],
            learning_rate=learning_rate,
            loss_func=LogLoss)
        
        self.assertEqual(len(network.layer_lst), 2)
        self.assertEqual(network.learning_rate, 0.1)
        self.assertIsInstance(network.layer_lst[0], DenseLayer)
        self.assertIsInstance(network.layer_lst[1], DenseLayer)
        self.assertIs(network.loss_func, LogLoss)
        self.assertIsNone(network.output_matrix)

    
    def test_forward(self):
        n_row_weight, n_col_weight = TestNeuralNetwork.layer2.weight_matrix.shape
        nonrandom_weight_matrix = np.arange(n_row_weight * n_col_weight)
        nonrandom_weight_matrix = np.reshape(nonrandom_weight_matrix, (n_row_weight, n_col_weight))
        TestNeuralNetwork.layer2.weight_matrix = nonrandom_weight_matrix
        self.assertTrue((TestNeuralNetwork.layer2.weight_matrix == np.array([
            [0, 1, 2],
            [3, 4, 5]
        ])).all())

        n_row_bias, n_col_bias = TestNeuralNetwork.layer2.bias_matrix.shape
        nonrandom_bias_matrix = np.arange(n_row_bias * n_col_bias)
        nonrandom_bias_matrix = -(nonrandom_bias_matrix + 10)
        nonrandom_bias_matrix = np.reshape(nonrandom_bias_matrix, (n_row_bias, n_col_bias))
        TestNeuralNetwork.layer2.bias_matrix = nonrandom_bias_matrix
        self.assertTrue((TestNeuralNetwork.layer2.bias_matrix == np.array([
            [[-10, -11, -12]]
        ])).all())

        n_row_weight, n_col_weight = TestNeuralNetwork.layer3.weight_matrix.shape
        nonrandom_weight_matrix = np.arange(n_row_weight * n_col_weight)
        nonrandom_weight_matrix -= 3
        nonrandom_weight_matrix = np.reshape(nonrandom_weight_matrix, (n_row_weight, n_col_weight))
        TestNeuralNetwork.layer3.weight_matrix = nonrandom_weight_matrix
        self.assertTrue(np.equal(TestNeuralNetwork.layer3.weight_matrix, np.array([
            [-3, -2],
            [-1, 0],
            [1, 2]
        ])).all())

        n_row_bias, n_col_bias = TestNeuralNetwork.layer3.bias_matrix.shape
        nonrandom_bias_matrix = np.arange(n_row_bias * n_col_bias)
        nonrandom_bias_matrix += 1
        nonrandom_bias_matrix = np.reshape(nonrandom_bias_matrix, (n_row_bias, n_col_bias))
        TestNeuralNetwork.layer3.bias_matrix = nonrandom_bias_matrix
        self.assertTrue((TestNeuralNetwork.layer3.bias_matrix == np.array([
            [[1, 2]]
        ])).all())

        learning_rate = 0.1
        network = neural_network.NeuralNetwork(
            layer_lst=[
                TestNeuralNetwork.layer2, 
                TestNeuralNetwork.layer3
            ], learning_rate=learning_rate
        )
    
        network.forward(input_matrix=TestNeuralNetwork.input_matrix)
        self.assertTrue((network.output_matrix == np.array([
            [[1, 2],
             [5, 16],
             [0, 34]]
        ])).all())
    
    def test_backward(self):
        input_matrix = np.array([
            [0.3, 0.6, 0.9, 1.1, 2, 0],
            [-0.9, 0.0, 2.9, -0.1, 3, 1],
            [3.3, 1.6, -5.9, 0.7, 7, 1],
            [-1.9, -0.2, 0.3, 0.5, 4, 0],
            [0.2, 0.2, 0.9, 1.1, 1, 0],
            [-0.9, 0.0, 3.9, -3.1, 7, 0],
            [3.3, 1.6, -5.9, 0.7, 1, 1],
            [-1.9, -9.2, 0.3, 0.5, 0, 1]])
        feature_matrix = input_matrix[:, :-1]
        feature_size, feature_dim = feature_matrix.shape
        label_matrix = input_matrix[:, -1]
        label_matrix = np.reshape(label_matrix, (-1, 1))
        n_neuron_lst = [feature_dim, 4, 3, 1]
        neural_net = neural_network.NeuralNetwork(
            layer_lst=[
                DenseLayer(input_dim=n_neuron_lst[0],
                    output_dim=n_neuron_lst[1],
                        activation_func=activation.ReLU),
                DenseLayer(input_dim=n_neuron_lst[1],
                    output_dim=n_neuron_lst[2],
                        activation_func=activation.ReLU),
                DenseLayer(input_dim=n_neuron_lst[2],
                    output_dim=n_neuron_lst[3],
                        activation_func=activation.Sigmoid)],
                learning_rate=0.1,
                loss_func=LogLoss)
        neural_net.forward(input_matrix=feature_matrix)
        neural_net.backward(label_matrix=label_matrix)

        for layer in neural_net.layer_lst:
            self.assertIsNotNone(layer.derivative_wrt_weight)
            actual_derivative_wrt_weight = layer.derivative_wrt_weight
            layer.increment_weight()
            plus_epsilon_output = neural_net.forward(input_matrix=feature_matrix)
            layer.decrement_weight()
            layer.decrement_weight()
            minus_epsilon_output = neural_net.forward(input_matrix=feature_matrix)
            
            desired_derivative_wrt_weight = numeric_derivative(
                function=LogLoss.function,
                variable_name="output",
                param_dict={"output": layer.weight_matrix, "actual": label_matrix})
            np.testing.assert_allclose(
                actual_derivative_wrt_weight,
                desired_derivative_wrt_weight, rtol=1e-4)
            layer.increment_weight()


if __name__ == "__main__": unittest.main(verbosity=3)

