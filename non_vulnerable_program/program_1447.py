from pyrex_ext.primes import primes
restore_path()

class test_primes(ScipyTestCase):
    def check_simple(self, level=1):
        l = primes(10)
        assert_equal(l, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
if __name__ == "__main__":
    ScipyTest().run()


#!/usr/bin/env python
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('testnumpydistutils',parent_package,top_path)
    config.add_subpackage('pyrex_ext')
    config.add_subpackage('f2py_ext')
    #config.add_subpackage('f2py_f90_ext')
    config.add_subpackage('swig_ext')
    config.add_subpackage('gen_ext')
    return config

if __name__ == "__main__":
