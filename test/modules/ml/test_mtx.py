import unittest, sys, os, logging
import numpy as np


FILE_DEPTH = 4
sys.path.append(
    "\\".join(os.path.abspath(__file__).split("\\")[:-FILE_DEPTH]))
from modules.ml.mtx import train_test_split


class TestMatrixHelper(unittest.TestCase):
    def test_train_test_split(self):
        arr = np.array( [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6],
            [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 10],
            [8, 9, 10, 11]])
        ftr, lbl = arr[:, :-1], arr[:, -1]
        train_ftr, test_ftr, train_lbl, test_lbl =\
            train_test_split(ftr, lbl, test_frac=0.49)
        self.assertEqual(len(train_ftr), len(train_lbl))
        self.assertEqual(len(train_ftr), 4)
        self.assertEqual(len(test_ftr), len(test_lbl))
        self.assertEqual(len(test_ftr), 4)
        train_ftr, test_ftr = train_ftr.T, test_ftr.T
        train_lbl, test_lbl = np.reshape(train_lbl, (1, -1)), \
            np.reshape(test_lbl, (1, -1))


if __name__ == "__main__":
    unittest.main(verbosity=3)
