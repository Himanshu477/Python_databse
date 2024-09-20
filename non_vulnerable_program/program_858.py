import scipy.base;reload(scipy.base)
from scipy.base import *
del sys.path[0]

##################################################
### Test for sum

class test_float(unittest.TestCase):
    def check_nothing(self):
        pass

class test_double(unittest.TestCase):
    def check_nothing(self):
        pass

if __name__ == "__main__":
    ScipyTest('scipy.base.limits').run()



""" Test functions for matrix module

"""

