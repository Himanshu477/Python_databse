import scipy.base;reload(scipy.base)
from scipy.base.getlimits import *
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
    ScipyTest().run()


def configuration(parent_package='',top_path=None):
    from scipy.distutils.misc_util import Configuration
    config = Configuration('basic',parent_package,top_path)
    config.add_data_dir('tests')
    return config


#!/usr/bin/env python
# Copied from fftpack.helper by Pearu Peterson, October 2005
""" Test functions for fftpack.helper module
"""

