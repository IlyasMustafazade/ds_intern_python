import unittest, sys, os, logging
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from copy import deepcopy

FILE_DEPTH = 4
sys.path.append(
    "\\".join(os.path.abspath(__file__).split("\\")[:-FILE_DEPTH]))
from modules.dl.optim import GD
from modules.dl.nn import NN
from modules.dl.loss import logloss, l2
from modules.dl.layer import Dense
from modules.dl.activ import relu, sigmoid

logging.basicConfig(level=logging.INFO)


class TestGD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        in_mtx = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
        in_size = len(in_mtx)
        cls.ftr = in_mtx[:, :-1].T
        cls.lbl = np.reshape(in_mtx[:, -1], (1, -1))
        cls.eta = 1
        cls.optimizer = GD(cls.eta)
        n_iter = 200
        cls.net = NN(len(cls.ftr), loss_fn=logloss,
            n_iter=n_iter, optimizer=cls.optimizer)
        n_neuron_lst = [4, 2, 1]
        cls.net.add([(Dense, relu), (Dense, relu), (Dense, sigmoid)],
            n_neuron_lst=n_neuron_lst)

    def test__call__(self):
        self.net.forward(self.ftr)
        self.net.backward(self.lbl)
        old_net = deepcopy(self.net)
        self.optimizer(self.net)
        for i in range(1, len(self.net.lyr_lst)):
            old_lyr, lyr = old_net.lyr_lst[i], self.net.lyr_lst[i]
            rtol = 1e-8
            assert_allclose(
                lyr.wgt, old_lyr.wgt - self.eta * old_lyr.d_wgt, rtol=rtol)
            assert_allclose(
                lyr.bias, old_lyr.bias - self.eta * old_lyr.d_bias, rtol=rtol)


if __name__ == "__main__":
    unittest.main(verbosity=3)
