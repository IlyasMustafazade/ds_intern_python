import unittest, sys, os
import numpy as np

FILE_DEPTH = 4
sys.path.append(
    "\\".join(os.path.abspath(__file__).split("\\")[:-FILE_DEPTH]))
from modules.testing.custom_assert import CustomAssert


class TestCustomAssert(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.custom_assert = CustomAssert()
    
    def test_assert_is_empty(self):
        self.custom_assert.assert_is_empty(np.empty((0,)))
        self.custom_assert.assert_is_empty([])
        self.custom_assert.assert_is_empty(tuple())
        with self.assertRaises(TypeError):
            self.custom_assert.assert_is_empty(0)

    def test_assert_is_not_empty(self):
        self.custom_assert.assert_is_not_empty(np.arange(1))
        self.custom_assert.assert_is_not_empty([1])
        self.custom_assert.assert_is_not_empty((1,))
        with self.assertRaises(TypeError):
            self.custom_assert.assert_is_not_empty(0)

    def test_assert_near_zero(self):
        UP_LIM = 1e-4
        self.custom_assert.assert_near_zero(1e-5, UP_LIM)
        self.custom_assert.assert_near_zero(0.001, 0.01)
        with self.assertRaises(AssertionError):
            self.custom_assert.assert_near_zero(-0.01, UP_LIM)

    def test_apply_assert(self):
        UP_LIM = 1e-4
        with self.assertRaises(AssertionError):
            self.custom_assert.apply_assert(
                [], self.custom_assert.assert_near_zero, (UP_LIM,))
        self.custom_assert.apply_assert(
            [0.01, 0.001], self.custom_assert.assert_near_zero, (0.1,))
        self.custom_assert.apply_assert(
            [0.1], self.custom_assert.assert_near_zero, (1,))
 

if __name__ == "__main__":
    unittest.main(verbosity=3)

