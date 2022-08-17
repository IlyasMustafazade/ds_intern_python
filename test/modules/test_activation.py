import unittest
import numpy as np
import sys
import os


FILE_DEPTH = 3
sys.path.append("\\".join(os.path.abspath(__file__).split("\\")[:-FILE_DEPTH]))
import modules.activation as activation


class TestReLU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix = np.array([
            [-12, 12, 23, -1],
            [-2, 11, 0, -12],
            [17, 19, 23, -33]
        ])

    def test_function(self):
        actual = activation.ReLU.function(TestReLU.matrix)
        desired = np.array([
            [0, 12, 23, 0],
            [0, 11, 0, 0],
            [17, 19, 23, 0]
        ])
        self.assertTrue(np.equal(actual, desired).all())
    
    def test_derivative(self):
        actual = activation.ReLU.derivative(TestReLU.matrix)
        desired = np.array([
            [0, 1, 1, 0],
            [0, 1, 0, 0],
            [1, 1, 1, 0]
        ])
        self.assertTrue(np.equal(actual, desired).all())


class TestSigmoid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix = np.array([
            [-1, 2],
            [3, -4]
        ])

    def test_function(self):
        actual = activation.Sigmoid.function(TestSigmoid.matrix)
        desired = np.array([
            [0.268941, 0.880797],
            [0.952574, 0.017986],
        ])
        np.testing.assert_allclose(actual, desired, rtol=1e-02)
    
    def test_derivative(self):
        actual = activation.Sigmoid.derivative(TestSigmoid.matrix)
        desired = np.array([
            [0.196366, 0.104993],
            [0.045176, 0.017662]
        ])
        np.testing.assert_allclose(actual, desired, rtol=1e-02)
    

if __name__ == "__main__": unittest.main(verbosity=3)

