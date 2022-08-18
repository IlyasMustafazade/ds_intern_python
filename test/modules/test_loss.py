import unittest
import numpy as np
import sys
import os


FILE_DEPTH = 3
sys.path.append("\\".join(os.path.abspath(__file__).split("\\")[:-FILE_DEPTH]))
import modules.loss as loss


class TestLoss(unittest.TestCase):
    output = np.array([
            [0.2, 0.1],
            [0.5, 0.6],
            [0.8, 0.5],
            [1.0, 0.3]
        ])
    actual = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0]
        ])

class TestL1(TestLoss):
    def test_function(self):
        output = np.array([
            [0.2],
            [0.5],
            [0.8],
            [1.0]
        ])
        actual = np.array([
            [0.0],
            [1],
            [1],
            [1]
        ])
        l1_loss = loss.L1.function(output=output, actual=actual)
        actual_loss = 0.225
        np.testing.assert_allclose(l1_loss, actual_loss, rtol=1e-08)

        l1_loss = loss.L1.function(output=TestLoss.output, actual=TestLoss.actual)
        actual_loss = 0.275
        np.testing.assert_allclose(l1_loss, actual_loss, rtol=1e-08)

    def test_derivative(self):
        pass


class TestL2(TestLoss):
    def test_function(self):
        l2_loss = loss.L2.function(output=TestLoss.output, actual=TestLoss.actual)
        actual_loss = 0.105
        np.testing.assert_allclose(l2_loss, actual_loss, rtol=1e-08)

    def test_derivative(self):
        l2_loss_derivative = loss.L2.derivative(output=TestLoss.output, actual=TestLoss.actual)
        actual_derivative = np.array([
            [0.4, 0.2],
            [-1.0, -0.8],
            [-0.4, -1.0],
            [0, 0.6]
        ])
        np.testing.assert_allclose(l2_loss_derivative, actual_derivative, rtol=1e-08)


class TestLogLoss(TestLoss):
    def test_function(self):
        log_loss = loss.LogLoss.function(output=TestLoss.output, actual=TestLoss.actual)
        actual_loss = 0.3507
        np.testing.assert_allclose(log_loss, actual_loss, rtol=1e-04)

    def test_derivative(self):
        log_loss_derivative = loss.LogLoss.derivative(output=TestLoss.output, actual=TestLoss.actual)
        actual_derivative = actual_derivative = np.array([
            [ 1.25,  1.11111111],
            [-2.  , -1.66666667],
            [-1.25, -2.        ],
            [ 0.  ,  1.42857143]
        ])
        np.testing.assert_allclose(log_loss_derivative, actual_derivative, rtol=1e-08)


if __name__ == "__main__": unittest.main(verbosity=3)
