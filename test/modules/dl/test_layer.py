import unittest, sys, os
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal


FILE_DEPTH = 4
sys.path.append(
    "\\".join(os.path.abspath(__file__).split("\\")[:-FILE_DEPTH]))
from modules.dl.layer import Input, Dense
from modules.dl.activ import identity, relu, sigmoid


class TestLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        in_size, in_dim = (2, 3)
        in_mtx = np.arange(in_size * in_dim)
        cls.in_mtx = np.reshape(in_mtx, (in_dim, in_size))
        cls.n_neuron = 2


class TestInput(TestLayer):
    def setUp(self):
        self.lyr = Input(self.n_neuron)

    def test_init(self):
        self.assertIsNone(self.lyr.output)
        self.assertIsNone(self.lyr.activ)
        self.assertEqual(self.lyr.out_dim, self.n_neuron)
        self.assertIs(self.lyr.activ_fn, identity)  

    def test_forward(self): 
        assert_array_equal(self.lyr.forward(self.in_mtx), self.in_mtx)


class TestDense(TestLayer):
    def setUp(self):
        self.lyr = Dense(len(self.in_mtx), self.n_neuron, relu)

    def test_init(self):
        self.assertEqual(self.lyr.out_dim, self.n_neuron)
        self.assertEqual(self.lyr.in_dim, len(self.in_mtx))
        self.assertIs(self.lyr.activ_fn, relu)
        self.assertIsNone(self.lyr.output)
        self.assertIsNone(self.lyr.activ)
        assert_array_equal(self.lyr.d_wgt, np.zeros(self.lyr.wgt.shape))
        assert_array_equal(self.lyr.d_bias, np.zeros(self.lyr.bias.shape))

    def test_forward(self):
        wgt = np.arange(self.lyr.wgt.size)
        wgt = np.reshape(wgt, self.lyr.wgt.shape)
        self.lyr.wgt = wgt
        np.testing.assert_array_equal(self.lyr.wgt, np.array([
            [0, 1], [2, 3], [4, 5]]))
        bias = -(np.arange(self.lyr.bias.size) + 30)
        bias = np.reshape(bias, self.lyr.bias.shape)
        self.lyr.bias = bias
        np.testing.assert_array_equal(self.lyr.bias, np.array([[-30], [-31]]))
        self.lyr.forward(self.in_mtx)
        np.testing.assert_array_equal(self.lyr.output, np.array([
            [-10, -4], [-5, 4]]))
        np.testing.assert_array_equal(self.lyr.activ, np.array([
            [0, 0], [0, 4]]))

    def test_set_d_wgt(self):
        d_wgt = np.ones(self.lyr.wgt.shape)
        self.lyr.set_d_wgt(d_wgt)
        assert_array_equal(self.lyr.d_wgt, np.ones(self.lyr.wgt.shape))
        with self.assertRaises(ValueError):
            self.lyr.set_d_wgt(np.ones((5, 6)))

    def test_set_d_bias(self):
        d_bias = np.ones(self.lyr.bias.shape)
        self.lyr.set_d_bias(d_bias)
        assert_array_equal(self.lyr.d_bias, np.ones(self.lyr.bias.shape))
        with self.assertRaises(ValueError):
            self.lyr.set_d_bias(np.ones((5, 6)))


if __name__ == "__main__":
    unittest.main(verbosity=3)
