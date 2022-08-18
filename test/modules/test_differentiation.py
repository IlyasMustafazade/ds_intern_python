import unittest
import numpy as np
import os
import sys


FILE_DEPTH = 3
sys.path.append("\\".join(os.path.abspath(__file__).split("\\")[:-FILE_DEPTH]))
import modules.differentiation as differentiation


class TestDifferentiation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        input_dim = 4 
        output_dim = 2
        weight_shape = (input_dim, output_dim)
        cls.matrix = 2 * np.random.rand(*weight_shape) - 1
    
    def function(X=None):
        return X * X

    def test_numeric_derivative(self):
        actual_derivative = differentiation.numeric_derivative(
            function=TestDifferentiation.function,
                variable_name="X",
                    param_dict={"X": TestDifferentiation.matrix})
        desired_derivative = 2 * TestDifferentiation.matrix
        np.testing.assert_allclose(actual_derivative, desired_derivative, rtol=1e-04)
    
    
if __name__ == "__main__": unittest.main(verbosity=3)

