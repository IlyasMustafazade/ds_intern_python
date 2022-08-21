import unittest, sys, os
import numpy as np
from numpy.testing import assert_allclose


FILE_DEPTH = 4
sys.path.append(
    "\\".join(os.path.abspath(__file__).split("\\")[:-FILE_DEPTH]))
from modules.dl.nn import NN
from modules.dl.layer import InputLayer, DenseLayer
from modules.dl.activ import relu, sigmoid
from modules.dl.loss import logloss, l2


class TestNN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        in_dim, in_size = (3, 2)
        in_mtx = np.arange(in_dim * in_size)
        cls.in_mtx = np.reshape(in_mtx, (in_dim, in_size))
        cls.n_neuron_lst = [in_dim, 3, 2]
        cls.lyr1 = DenseLayer(cls.n_neuron_lst[0], cls.n_neuron_lst[1], relu)
        cls.lyr2 = DenseLayer(cls.n_neuron_lst[1], cls.n_neuron_lst[2], relu)
        cls.EPSILON = 1e-4

    def test__init__(self):
        learning_rate = 0.1
        net = NN(len(self.in_mtx),
            learning_rate=learning_rate, loss_func=logloss)
        net.add_multi([(DenseLayer, self.n_neuron_lst[1], relu),
                       (DenseLayer, self.n_neuron_lst[2], relu)])
        self.assertEqual(len(net.layer_lst), 3)
        self.assertEqual(net.layer_lst[0].out_dim, 3)
        self.assertEqual(net.learning_rate, 0.1)
        self.assertIsInstance(net.layer_lst[0], InputLayer)
        for layer in net.layer_lst[1:]:
            self.assertIsInstance(layer, DenseLayer)
        self.assertIs(net.loss_func, logloss)
        self.assertIsNone(net.output)

    def test_forward(self):
        wgt = np.arange(self.lyr1.wgt.size)
        wgt = np.reshape(wgt, self.lyr1.wgt.shape)
        self.lyr1.wgt = wgt
        assert_allclose(
            self.lyr1.wgt, np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
        bias = -(np.arange(self.lyr1.bias.size) + 10)
        bias = np.reshape(bias, self.lyr1.bias.shape)
        self.lyr1.bias = bias
        assert_allclose(self.lyr1.bias, np.array([[-10], [-11], [-12]]))
        wgt = np.arange(self.lyr2.wgt.size) - 3
        wgt = np.reshape(wgt, self.lyr2.wgt.shape)
        self.lyr2.wgt = wgt
        assert_allclose(self.lyr2.wgt, np.array([[-3, -2], [-1, 0], [1, 2]]))
        bias = np.arange(self.lyr2.bias.size) + 1
        bias = np.reshape(bias, self.lyr2.bias.shape)
        self.lyr2.bias = bias
        assert_allclose(self.lyr2.bias, np.array([[1], [2]]))
        learning_rate = 0.1
        net = NN(len(self.in_mtx),
            learning_rate=learning_rate, loss_func=logloss)
        net.layer_lst.append(self.lyr1)
        net.layer_lst.append(self.lyr2)
        net.forward(self.in_mtx)
        assert_allclose(net.output, np.array([[0, 0], [22, 34]]), rtol=1e-4)

    def test_backward(self):
        in_mtx = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
        ftr = in_mtx[:, :-1]
        ftr = ftr.T
        label = in_mtx[:, -1]
        label = np.reshape(label, (1, -1))
        n_neuron_lst = [4, 1]
        net = NN(len(ftr), learning_rate=0.1, loss_func=l2)
        net.add_multi([(DenseLayer, n_neuron_lst[0], relu),
            (DenseLayer, n_neuron_lst[1], sigmoid)])
        net.forward(ftr)
        net.backward(label)

        for layer in net.layer_lst[1:]:
            self.assertIsNotNone(layer.deriv_wrt_wgt)
            actual_deriv_wrt_wgt = layer.deriv_wrt_wgt
            n_row, n_col = layer.wgt.shape
            for i in range(n_row):
                for j in range(n_col):
                    layer.wgt[i][j] = layer.wgt[i][j] + self.EPSILON
                    net.forward(ftr)
                    plus_epsilon_out = net.output
                    layer.wgt[i][j] = layer.wgt[i][j] - (2 * self.EPSILON)
                    net.forward(ftr)
                    minus_epsilon_out = net.output
                    num_deriv = (net.loss_func(
                        output=plus_epsilon_out, actual=label)
                        - net.loss_func(output=minus_epsilon_out,
                            actual=label)) / (2 * self.EPSILON)
                    backprop_deriv = layer.deriv_wrt_wgt[i][j]
                    assert_allclose(num_deriv, backprop_deriv, rtol=1e-8)
                    layer.wgt[i][j] += self.EPSILON
    
    def test_fit(self):
        in_mtx = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
        ftr = in_mtx[:, :-1]
        ftr = ftr.T
        label = in_mtx[:, -1]
        label = np.reshape(label, (1, -1))
        learning_rate = 1
        net = NN(len(ftr), learning_rate=learning_rate, loss_func=logloss)
        n_neuron_lst = [16, 8, 1]
        net.add_multi([(DenseLayer, n_neuron_lst[0], sigmoid),
                       (DenseLayer, n_neuron_lst[1], sigmoid),
                       (DenseLayer, n_neuron_lst[2], sigmoid)])
        n_epoch = 1000

        for epoch in range(n_epoch):
            net.forward(ftr)
            loss_ = net.loss_func(output=net.output, actual=label)
            if epoch % (n_epoch // 10) == 0: 
                print(f"Epoch: {epoch} | loss: {loss_}")
            net.backward(label)
            for layer in net.layer_lst[1:]:
                layer.wgt = layer.wgt - learning_rate * layer.deriv_wrt_wgt


if __name__ == "__main__":
    unittest.main(verbosity=3)
