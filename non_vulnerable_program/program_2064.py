from numpy.testing import *
import numpy.lib;reload(numpy.lib)
from numpy.lib.getlimits import finfo, iinfo
from numpy import single,double,longdouble
import numpy as np

##################################################

class TestPythonFloat(NumpyTestCase):
    def check_singleton(self):
        ftype = finfo(float)
        ftype2 = finfo(float)
        assert_equal(id(ftype),id(ftype2))

class TestSingle(NumpyTestCase):
    def check_singleton(self):
        ftype = finfo(single)
        ftype2 = finfo(single)
        assert_equal(id(ftype),id(ftype2))

class TestDouble(NumpyTestCase):
    def check_singleton(self):
        ftype = finfo(double)
        ftype2 = finfo(double)
        assert_equal(id(ftype),id(ftype2))

class TestLongdouble(NumpyTestCase):
    def check_singleton(self,level=2):
        ftype = finfo(longdouble)
        ftype2 = finfo(longdouble)
        assert_equal(id(ftype),id(ftype2))

class TestIinfo(NumpyTestCase):
    def check_basic(self):
        dts = zip(['i1', 'i2', 'i4', 'i8',
                   'u1', 'u2', 'u4', 'u8'],
                  [np.int8, np.int16, np.int32, np.int64,
                   np.uint8, np.uint16, np.uint32, np.uint64])
        for dt1, dt2 in dts:
            assert_equal(iinfo(dt1).min, iinfo(dt2).min)
            assert_equal(iinfo(dt1).max, iinfo(dt2).max)
        self.assertRaises(ValueError, iinfo, 'f4')

    def check_unsigned_max(self):
        types = np.sctypes['uint']
        for T in types:
            assert_equal(iinfo(T).max, T(-1))

if __name__ == "__main__":
    NumpyTest().run()


"""
>>> import numpy.core as nx
>>> from numpy.lib.polynomial import poly1d, polydiv

>>> p = poly1d([1.,2,3])
>>> p
poly1d([ 1.,  2.,  3.])
>>> print p
   2
1 x + 2 x + 3
>>> q = poly1d([3.,2,1])
>>> q
poly1d([ 3.,  2.,  1.])
>>> print q
   2
3 x + 2 x + 1

>>> p(0)
3.0
>>> p(5)
38.0
>>> q(0)
1.0
>>> q(5)
86.0

>>> p * q
poly1d([  3.,   8.,  14.,   8.,   3.])
>>> p / q
(poly1d([ 0.33333333]), poly1d([ 1.33333333,  2.66666667]))
>>> p + q
poly1d([ 4.,  4.,  4.])
>>> p - q
poly1d([-2.,  0.,  2.])
>>> p ** 4
poly1d([   1.,    8.,   36.,  104.,  214.,  312.,  324.,  216.,   81.])

>>> p(q)
poly1d([  9.,  12.,  16.,   8.,   6.])
>>> q(p)
poly1d([  3.,  12.,  32.,  40.,  34.])

>>> nx.asarray(p)
array([ 1.,  2.,  3.])
>>> len(p)
2

>>> p[0], p[1], p[2], p[3]
(3.0, 2.0, 1.0, 0)

>>> p.integ()
poly1d([ 0.33333333,  1.        ,  3.        ,  0.        ])
>>> p.integ(1)
poly1d([ 0.33333333,  1.        ,  3.        ,  0.        ])
>>> p.integ(5)
poly1d([ 0.00039683,  0.00277778,  0.025     ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ])
>>> p.deriv()
poly1d([ 2.,  2.])
>>> p.deriv(2)
poly1d([ 2.])

>>> q = poly1d([1.,2,3], variable='y')
>>> print q
   2
1 y + 2 y + 3
>>> q = poly1d([1.,2,3], variable='lambda')
>>> print q
        2
1 lambda + 2 lambda + 3

>>> polydiv(poly1d([1,0,-1]), poly1d([1,1]))
(poly1d([ 1., -1.]), poly1d([ 0.]))
"""

