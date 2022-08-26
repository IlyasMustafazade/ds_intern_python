import unittest, sys, os, logging
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal


FILE_DEPTH = 4
sys.path.append(
    "\\".join(os.path.abspath(__file__).split("\\")[:-FILE_DEPTH]))
from modules.dl.nn import NN, History
from modules.dl.layer import Input, Dense
from modules.dl.activ import relu, sigmoid, identity
from modules.dl.loss import logloss, l2
from modules.dl.optim import GD
from modules.ml.mtx import train_test_split
from modules.testing.custom_assert import CustomAssert


class TestNN(CustomAssert):
    @classmethod
    def setUpClass(cls):
        in_dim, in_size = (3, 2)
        in_mtx = np.arange(in_dim * in_size)
        cls.in_mtx = np.reshape(in_mtx, (in_dim, in_size))
        cls.n_neuron_lst = [in_dim, 3, 2]
        cls.lyr1 = Dense(cls.n_neuron_lst[0], cls.n_neuron_lst[1], relu)
        cls.lyr2 = Dense(cls.n_neuron_lst[1], cls.n_neuron_lst[2], relu)
        cls.XOR = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])

    def test__init__(self):
        net = NN(len(self.in_mtx), logloss)
        net.add([(Dense, relu), (Dense, relu)], n_neuron_lst=self.n_neuron_lst)
        self.assertEqual(len(net.lyr_lst), 3)
        self.assertEqual(net.lyr_lst[0].out_dim, 3)
        self.assertIsInstance(net.lyr_lst[0], Input)
        self.assertIs(net.loss_fn, logloss)
        for lyr in net.lyr_lst[1:]:
            self.assertIsInstance(lyr, Dense)
        none_lst = [net.output, net.history, net.n_iter, net.optimizer]
        for attr in none_lst:
            self.assertIsNone(attr)


    def test_forward(self):
        wgt = np.arange(self.lyr1.wgt.size)
        self.lyr1.wgt = np.reshape(wgt, self.lyr1.wgt.shape)
        assert_array_equal(
            self.lyr1.wgt, np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
        bias = -(np.arange(self.lyr1.bias.size) + 10)
        self.lyr1.bias = np.reshape(bias, self.lyr1.bias.shape)
        assert_array_equal(self.lyr1.bias, np.array([[-10], [-11], [-12]]))
        wgt = np.arange(self.lyr2.wgt.size) - 3
        self.lyr2.wgt = np.reshape(wgt, self.lyr2.wgt.shape)
        assert_array_equal(
            self.lyr2.wgt, np.array([[-3, -2], [-1, 0], [1, 2]]))
        bias = np.arange(self.lyr2.bias.size) + 1
        self.lyr2.bias = np.reshape(bias, self.lyr2.bias.shape)
        assert_array_equal(self.lyr2.bias, np.array([[1], [2]]))
        net = NN(len(self.in_mtx), loss_fn=logloss)
        net.lyr_lst.append(self.lyr1)
        net.lyr_lst.append(self.lyr2)
        net.forward(self.in_mtx)
        assert_array_equal(net.output, np.array([[0, 0], [22, 34]]))

    def test_backward(self):
        in_mtx = self.XOR
        ftr = in_mtx[:, :-1].T
        lbl = np.reshape(in_mtx[:, -1], (1, -1))
        n_neuron_lst = [16, 4, 1]
        net = NN(len(ftr), loss_fn=l2)
        net.add([(Dense, relu), (Dense, relu), (Dense, sigmoid)],
            n_neuron_lst=n_neuron_lst)
        net.forward(ftr)
        net.backward(lbl)
        self.assert_num_equal_backprop(net, ftr, lbl)

    def assert_num_equal_backprop(self, net, ftr, lbl):
        d_wgt_lst, d_bias_lst = NumGrad(net, ftr, lbl)()
        rtol = 1e-4
        for i, lyr in enumerate(net.lyr_lst[1:]):
            assert_allclose(lyr.d_wgt, d_wgt_lst[i], rtol=rtol)
            assert_allclose(lyr.d_bias, d_bias_lst[i], rtol=rtol)

    def make_history(self):
        in_mtx = self.XOR
        ftr, lbl = in_mtx[:, :-1].T, np.reshape(in_mtx[:, -1], (1, -1))
        eta, n_iter = 2, 4096
        net = NN(len(ftr), optimizer=GD(eta), loss_fn=logloss, n_iter=n_iter)
        n_neuron_lst = [32, 32, 1]
        net.add([(Dense, relu), (Dense, sigmoid),
            (Dense, sigmoid)], n_neuron_lst=n_neuron_lst)
        net.fit(ftr, ftr, lbl, lbl)
        return net.history

    def test_fit(self):
        UP_LIM = 1e-4
        num_iter = 16
        for i in range(num_iter):
            history = self.make_history()
            self.assert_near_zero(history.train_loss[-1], UP_LIM)


class TestHistory(CustomAssert):
    @classmethod
    def setUpClass(cls):
        cls.history = History()

    def test_record(self):
        self.assert_is_empty(self.history.train_loss)
        self.assert_is_empty(self.history.test_loss)
        self.history.record("train", np.array([1, 2, 3, 4, 5]))
        assert_array_equal(self.history.train_loss, np.array([1, 2, 3, 4, 5]))
        self.history.record("train", np.array([6, 7, 8, 9, 10]))
        assert_array_equal(
            self.history.train_loss, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        self.history.record("test", 0)
        assert_array_equal(self.history.test_loss, np.array([0]))
        self.history.record("test", np.array([1, 2]))
        assert_array_equal(self.history.test_loss, np.array([0, 1, 2]))


class NumGrad:
    def __init__(self, net, ftr, lbl):
        self.lyr_lst = net.lyr_lst
        self.forward = net.forward
        self.loss_fn = net.loss_fn
        self.ftr = ftr
        self.lbl = lbl
        self.EPSILON = 1e-4

    def __call__(self):
        d_wgt_lst, d_bias_lst = [], []
        for lyr in self.lyr_lst[1:]:
            d_wgt_lst.append(self.__lyr_deriv(lyr.wgt))
            d_bias_lst.append(self.__lyr_deriv(lyr.bias))
        return d_wgt_lst, d_bias_lst

    def __lyr_deriv(self, param):
        d_lyr = np.zeros(param.shape)
        for i in range(len(param)):
            d_lyr[i] = self.__row_deriv(i, param)
        return d_lyr

    def __row_deriv(self, row_idx, param):
        n_col = len(param.T)
        d_row = np.zeros(n_col)
        for i in range(n_col):
            param[row_idx][i] += self.EPSILON
            plus_out = self.forward(self.ftr)
            param[row_idx][i] -= 2 * self.EPSILON
            minus_out = self.forward(self.ftr)
            d_row[i] = (self.loss_fn(plus_out, self.lbl) - self.loss_fn(minus_out, self.lbl)) / (2 * self.EPSILON)
            param[row_idx][i] += self.EPSILON
        return d_row


if __name__ == "__main__":
    unittest.main(verbosity=3)
