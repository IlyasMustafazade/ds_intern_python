import unittest, sys, os, logging
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal


FILE_DEPTH = 4
sys.path.append(
    "\\".join(os.path.abspath(__file__).split("\\")[:-FILE_DEPTH]))
from modules.dl.loss import *


class TestLoss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = np.array([[0.2, 0.1], [0.5, 0.6], [0.8, 0.5], [1.0, 0.3]])
        cls.act = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [1.0, 0.0]])


class TestL1(TestLoss):
    def test_l1(self):
        output = np.array([[0.2], [0.5], [0.8], [1.0]])
        act = np.array([[0.0], [1], [1], [1]])
        act_loss = 0.225
        assert_allclose(l1(output, act), act_loss, rtol=1e-08)
        act_loss = 0.275
        assert_allclose(l1(self.output, self.act), act_loss, rtol=1e-08)

    def test_l1_deriv(self):
        pass


class TestL2(TestLoss):
    def test_l2(self):
        act_loss = 0.105
        assert_allclose(l2(self.output, self.act), act_loss, rtol=1e-08)

    def test_l2_deriv(self):
        act_deriv = np.array(
            [[0.4, 0.2], [-1.0, -0.8], [-0.4, -1.0], [0, 0.6]]) / 8
        assert_allclose(l2_deriv(self.output, self.act), act_deriv, rtol=1e-08)
        assert_allclose(l2(self.output, self.act, deriv=True),
            act_deriv, rtol=1e-08)


class TestLogLoss(TestLoss):
    def test_logloss(self):
        log_loss = logloss(self.output, self.act)
        act_loss = 0.3507
        assert_allclose(log_loss, act_loss, rtol=1e-04)

    def test_logloss_deriv(self):
        log_loss_deriv = logloss_deriv(self.output, self.act)
        act_deriv = np.array([[1.25, 1.11111111], [-2, -1.66666667],
            [-1.25, -2], [0, 1.42857143]]) / 8
        assert_allclose(log_loss_deriv, act_deriv, rtol=1e-08)
        assert_allclose(logloss(self.output, self.act, deriv=True),
            act_deriv, rtol=1e-08)


if __name__ == "__main__":
    unittest.main(verbosity=3)
