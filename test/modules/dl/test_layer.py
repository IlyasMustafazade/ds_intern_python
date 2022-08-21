import unittest, sys, os
import numpy as np
from numpy.testing import assert_allclose


FILE_DEPTH = 4
sys.path.append(
    "\\".join(os.path.abspath(__file__).split("\\")[:-FILE_DEPTH]))
from modules.dl.layer import InputLayer, DenseLayer
from modules.dl.activ import identity, relu, sigmoid


class TestLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        in_size, in_dim = (2, 3)
        in_mtx = np.arange(in_size * in_dim)
        cls.in_mtx = np.reshape(in_mtx, (in_dim, in_size))
        cls.n_neuron = 2


class TestInputLayer(TestLayer):
    def setUp(self):
        self.lyr = InputLayer(len(self.in_mtx.T))
    
    def test_init(self):
        self.assertIsNone(self.lyr.output)
        self.assertIsNone(self.lyr.output)
        assert_allclose(self.lyr.out_dim, len(self.in_mtx.T))
        self.assertIs(self.lyr.activ_func, identity)  

    def test_forward(self): 
        output = self.lyr.forward(self.in_mtx)
        des = self.in_mtx
        assert_allclose(output, des, rtol=1e-8)
        

class TestDenseLayer(TestLayer):
    def setUp(self):
        self.lyr = DenseLayer(len(self.in_mtx), self.n_neuron, relu)

    def test_init(self):            
        self.assertEqual(self.lyr.out_dim, self.n_neuron)
        self.assertEqual(self.lyr.in_dim, len(self.in_mtx))
        self.assertIs(self.lyr.activ_func, relu)
        none_attr_lst = [self.lyr.output, self.lyr.activ]
        for attr in none_attr_lst:
            self.assertIsNone(attr)
        assert_allclose(self.lyr.deriv_wrt_wgt,
            np.zeros(self.lyr.wgt.shape), rtol=1e-8)
        assert_allclose(self.lyr.deriv_wrt_bias,
            np.zeros(self.lyr.bias.shape), rtol=1e-8)

    def test_forward(self):
        wgt = np.arange(self.lyr.wgt.size)
        wgt = np.reshape(wgt, self.lyr.wgt.shape)
        self.lyr.wgt = wgt
        np.testing.assert_allclose(self.lyr.wgt, np.array([
            [0, 1], [2, 3], [4, 5]]), rtol=1e-8)
        bias = np.arange(self.lyr.bias.size)
        bias += 30
        bias = np.reshape(bias, self.lyr.bias.shape)
        self.lyr.bias = -bias
        np.testing.assert_allclose(self.lyr.bias, np.array(
            [[-30], [-31]]), rtol=1e-8)
        self.lyr.forward(self.in_mtx)
        np.testing.assert_allclose(self.lyr.output, np.array([
            [-10, -4], [-5, 4]]), rtol=1e-8)
        np.testing.assert_allclose(self.lyr.activ, np.array([
            [0, 0], [0, 4]]), rtol=1e-8)
        
    def test_set_deriv_wrt_wgt(self):
        deriv_wrt_wgt = np.ones(self.lyr.wgt.shape)
        self.lyr.set_deriv_wrt_wgt(deriv_wrt_wgt)
        assert_allclose(self.lyr.deriv_wrt_wgt,
            np.ones(self.lyr.wgt.shape), rtol=1e-8)
        with self.assertRaises(ValueError):
            self.lyr.set_deriv_wrt_wgt(np.ones((5, 6)))
    
    def test_set_deriv_wrt_bias(self):
        deriv_wrt_bias = np.ones(self.lyr.bias.shape)
        self.lyr.set_deriv_wrt_bias(deriv_wrt_bias)
        assert_allclose(self.lyr.deriv_wrt_bias,
            np.ones(self.lyr.bias.shape), rtol=1e-8)
        with self.assertRaises(ValueError):
            self.lyr.set_deriv_wrt_bias(np.ones((5, 6)))

    
if __name__ == "__main__":
    unittest.main(verbosity=3)
