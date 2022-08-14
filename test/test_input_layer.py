import unittest
import numpy as np
import sys
import os


FILE_DEPTH = 2
sys.path.append("\\".join(os.path.abspath(__file__).split("\\")[:-FILE_DEPTH]))
import modules.input_layer as input_layer


class TestInputLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input_size, cls.input_dim = (20, 2)
        input_matrix = np.arange(cls.input_size * cls.input_dim)
        cls.input_matrix = np.reshape(input_matrix, (cls.input_size, cls.input_dim))


    def test_forward_pass(self):
        layer = input_layer.InputLayer(output_dim=TestInputLayer.input_dim)
        self.assertEqual(layer.output_dim, TestInputLayer.input_dim)
        self.assertIsNone(layer.activation_matrix)
        layer.forward_pass(input_matrix=TestInputLayer.input_matrix)
        self.assertTrue((layer.activation_matrix == TestInputLayer.input_matrix).all())
        print(layer)


if __name__ == "__main__": unittest.main(verbosity=3)
