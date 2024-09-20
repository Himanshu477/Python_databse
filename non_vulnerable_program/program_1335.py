import numpy.base;reload(numpy.base)
from numpy.base import *
restore_path()

class test_grid(ScipyTestCase):
    def check_basic(self):
        a = mgrid[-1:1:10j]
        b = mgrid[-1:1:0.1]
        assert(a.shape == (10,))
        assert(b.shape == (20,))
        assert(a[0] == -1)
        assert_almost_equal(a[-1],1)
        assert(b[0] == -1)
        assert_almost_equal(b[1]-b[0],0.1,11)
        assert_almost_equal(b[-1],b[0]+19*0.1,11)
        assert_almost_equal(a[1]-a[0],2.0/9.0,11)

    def check_nd(self):
        c = mgrid[-1:1:10j,-2:2:10j]
        d = mgrid[-1:1:0.1,-2:2:0.2]
        assert(c.shape == (2,10,10))
        assert(d.shape == (2,20,20))
        assert_array_equal(c[0][0,:],-ones(10,'d'))
        assert_array_equal(c[1][:,0],-2*ones(10,'d'))
        assert_array_almost_equal(c[0][-1,:],ones(10,'d'),11)
        assert_array_almost_equal(c[1][:,-1],2*ones(10,'d'),11)
        assert_array_almost_equal(d[0,1,:]-d[0,0,:], 0.1*ones(20,'d'),11)
        assert_array_almost_equal(d[1,:,1]-d[1,:,0], 0.2*ones(20,'d'),11)

class test_concatenator(ScipyTestCase):
    def check_1d(self):
        assert_array_equal(r_[1,2,3,4,5,6],array([1,2,3,4,5,6]))
        b = ones(5)
        c = r_[b,0,0,b]
        assert_array_equal(c,[1,1,1,1,1,0,0,1,1,1,1,1])

    def check_2d(self):
        b = rand(5,5)
        c = rand(5,5)
        d = r_[b,c,'1']  # append columns
        assert(d.shape == (5,10))
        assert_array_equal(d[:,:5],b)
        assert_array_equal(d[:,5:],c)
        d = r_[b,c]
        assert(d.shape == (10,5))
        assert_array_equal(d[:5,:],b)
        assert_array_equal(d[5:,:],c)

if __name__ == "__main__":
    ScipyTest().run()


"""
>>> import numpy.base as nx
>>> from numpy.base.polynomial import poly1d, polydiv

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

