import unittest, logging


class CustomAssert(unittest.TestCase):
    def assert_is_empty(self, iter_):
        self.assertEqual(len(iter_), 0)
    
    def assert_is_not_empty(self, iter_):
        self.assertNotEqual(len(iter_), 0)
    
    def assert_near_zero(self, val, upper_lim):
        assert(val > 0)
        self.assertTrue(val < upper_lim)
    
    def apply_assert(self, iter_, assert_, args):
        assert (len(iter_) > 0)
        for elem in iter_:
            assert_(elem, *args)
