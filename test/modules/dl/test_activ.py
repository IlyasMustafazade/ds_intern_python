import unittest, sys, os, logging
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal


FILE_DEPTH = 4
sys.path.append(
    "\\".join(os.path.abspath(__file__).split("\\")[:-FILE_DEPTH]))
from modules.dl.activ import *


class TestIdentity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mtx = np.array([[1, 2], [2, 3]])
    
    def test_identity(self):
        assert_array_equal(identity(self.mtx), self.mtx)

    def test_identity_deriv(self):
        assert_array_equal(
            identity(self.mtx, deriv=True), np.ones(self.mtx.shape))


class TestReLU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mtx = np.array([[-12, 12, 23, -1],
                            [-2, 11, 0, -12],
                            [17, 19, 23, -33]])

    def test_relu(self):
        des = np.array([[0, 12, 23, 0], [0, 11, 0, 0], [17, 19, 23, 0]])
        assert_array_equal(relu(self.mtx), des)
    
    def test_relu_deriv(self):
        des = np.array([[0, 1, 1, 0], [0, 1, 0, 0], [1, 1, 1, 0]])
        assert_array_equal(relu_deriv(self.mtx), des)
        assert_array_equal(relu(self.mtx, deriv=True), des)


class TestSigmoid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mtx = np.array([[-1, 2], [3, -4]])

    def test_sigmoid(self):
        des = np.array([[0.268941, 0.880797], [0.952574, 0.017986]])
        assert_allclose(sigmoid(self.mtx), des, rtol=1e-02)
    
    def test_sigmoid_deriv(self):
        des = np.array([[0.196366, 0.104993], [0.045176, 0.017662]])
        assert_allclose(sigmoid_deriv(self.mtx), des, rtol=1e-02)
        assert_allclose(sigmoid(self.mtx, deriv=True), des, rtol=1e-02)
    

if __name__ == "__main__":
    unittest.main(verbosity=3)
