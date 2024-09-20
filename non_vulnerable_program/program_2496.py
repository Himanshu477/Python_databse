import numpy.polynomial as poly
from numpy.testing import TestCase, run_module_suite, assert_

class test_str(TestCase):
    def test_polynomial_str(self):
        res = str(poly.Polynomial([0,1]))
        tgt = 'poly([0., 1.])'
        assert_(res, tgt)


    def test_chebyshev_str(self):
        res = str(poly.Chebyshev([0,1]))
        tgt = 'leg([0., 1.])'
        assert_(res, tgt)


    def test_legendre_str(self):
        res = str(poly.Legendre([0,1]))
        tgt = 'leg([0., 1.])'
        assert_(res, tgt)


    def test_hermite_str(self):
        res = str(poly.Hermite([0,1]))
        tgt = 'herm([0., 1.])'
        assert_(res, tgt)


    def test_hermiteE_str(self):
        res = str(poly.HermiteE([0,1]))
        tgt = 'herme([0., 1.])'
        assert_(res, tgt)


    def test_laguerre_str(self):
        res = str(poly.Laguerre([0,1]))
        tgt = 'lag([0., 1.])'
        assert_(res, tgt)


class test_repr(TestCase):
    def test_polynomial_str(self):
        res = repr(poly.Polynomial([0,1]))
        tgt = 'Polynomial([0., 1.])'
        assert_(res, tgt)


    def test_chebyshev_str(self):
        res = repr(poly.Chebyshev([0,1]))
        tgt = 'Chebyshev([0., 1.], [-1., 1.], [-1., 1.])'
        assert_(res, tgt)


    def test_legendre_repr(self):
        res = repr(poly.Legendre([0,1]))
        tgt = 'Legendre([0., 1.], [-1., 1.], [-1., 1.])'
        assert_(res, tgt)


    def test_hermite_repr(self):
        res = repr(poly.Hermite([0,1]))
        tgt = 'Hermite([0., 1.], [-1., 1.], [-1., 1.])'
        assert_(res, tgt)


    def test_hermiteE_repr(self):
        res = repr(poly.HermiteE([0,1]))
        tgt = 'HermiteE([0., 1.], [-1., 1.], [-1., 1.])'
        assert_(res, tgt)


    def test_laguerre_repr(self):
        res = repr(poly.Laguerre([0,1]))
        tgt = 'Laguerre([0., 1.], [0., 1.], [0., 1.])'
        assert_(res, tgt)


#

if __name__ == "__main__":
    run_module_suite()


""" Doctests for NumPy-specific nose/doctest modifications
"""
# try the #random directive on the output line
def check_random_directive():
    '''
    >>> 2+2
    <BadExample object at 0x084D05AC>  #random: may vary on your system
    '''

# check the implicit "import numpy as np"
def check_implicit_np():
    '''
    >>> np.array([1,2,3])
    array([1, 2, 3])
    '''

# there's some extraneous whitespace around the correct responses
def check_whitespace_enabled():
    '''
    # whitespace after the 3
    >>> 1+2
    3

    # whitespace before the 7
    >>> 3+4
     7
    '''


if __name__ == '__main__':
    # Run tests outside numpy test rig
