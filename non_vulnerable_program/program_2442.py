import numpy as np
import numpy.matlib
from numpy.testing import assert_array_equal, assert_

def test_empty():
    x = np.matlib.empty((2,))
    assert_(isinstance(x, np.matrix))
    assert_(x.shape, (1,2))

def test_ones():
    assert_array_equal(np.matlib.ones((2, 3)),
                       np.matrix([[ 1.,  1.,  1.],
                                 [ 1.,  1.,  1.]]))

    assert_array_equal(np.matlib.ones(2), np.matrix([[ 1.,  1.]]))

def test_zeros():
    assert_array_equal(np.matlib.zeros((2, 3)),
                       np.matrix([[ 0.,  0.,  0.],
                                 [ 0.,  0.,  0.]]))

    assert_array_equal(np.matlib.zeros(2), np.matrix([[ 0.,  0.]]))

def test_identity():
    x = np.matlib.identity(2, dtype=np.int)
    assert_array_equal(x, np.matrix([[1, 0], [0, 1]]))

def test_eye():
    x = np.matlib.eye(3, k=1, dtype=int)
    assert_array_equal(x, np.matrix([[ 0,  1,  0],
                                     [ 0,  0,  1],
                                     [ 0,  0,  0]]))

def test_rand():
    x = np.matlib.rand(3)
    # check matrix type, array would have shape (3,)
    assert_(x.ndim == 2)

def test_randn():
    x = np.matlib.randn(3)
    # check matrix type, array would have shape (3,)
    assert_(x.ndim == 2)

def test_repmat():
    a1 = np.arange(4)
    x = np.matlib.repmat(a1, 2, 2)
    y = np.array([[0, 1, 2, 3, 0, 1, 2, 3],
                  [0, 1, 2, 3, 0, 1, 2, 3]])
    assert_array_equal(x, y)


if __name__ == "__main__":
    run_module_suite()


"""
Objects for dealing with Legendre series.

This module provides a number of objects (mostly functions) useful for
dealing with Legendre series, including a `Legendre` class that
encapsulates the usual arithmetic operations.  (General information
on how this module represents and works with such polynomials is in the
docstring for its "parent" sub-package, `numpy.polynomial`).

Constants
---------
- `legdomain` -- Legendre series default domain, [-1,1].
- `legzero` -- Legendre series that evaluates identically to 0.
- `legone` -- Legendre series that evaluates identically to 1.
- `legx` -- Legendre series for the identity map, ``f(x) = x``.

Arithmetic
----------
- `legmulx` -- multiply a Legendre series in ``P_i(x)`` by ``x``.
- `legadd` -- add two Legendre series.
- `legsub` -- subtract one Legendre series from another.
- `legmul` -- multiply two Legendre series.
- `legdiv` -- divide one Legendre series by another.
- `legval` -- evaluate a Legendre series at given points.

Calculus
--------
- `legder` -- differentiate a Legendre series.
- `legint` -- integrate a Legendre series.

Misc Functions
--------------
- `legfromroots` -- create a Legendre series with specified roots.
- `legroots` -- find the roots of a Legendre series.
- `legvander` -- Vandermonde-like matrix for Legendre polynomials.
- `legfit` -- least-squares fit returning a Legendre series.
- `legtrim` -- trim leading coefficients from a Legendre series.
- `legline` -- Legendre series of given straight line.
- `leg2poly` -- convert a Legendre series to a polynomial.
- `poly2leg` -- convert a polynomial to a Legendre series.

Classes
-------
- `Legendre` -- A Legendre series class.

See also
--------
`numpy.polynomial`

Notes
-----
The implementations of multiplication, division, integration, and
differentiation use the algebraic identities [1]_:

.. math ::
    T_n(x) = \\frac{z^n + z^{-n}}{2} \\\\
    z\\frac{dx}{dz} = \\frac{z - z^{-1}}{2}.

where

.. math :: x = \\frac{z + z^{-1}}{2}.

These identities allow a Chebyshev series to be expressed as a finite,
symmetric Laurent series.  In this module, this sort of Laurent series
is referred to as a "z-series."

References
----------
.. [1] A. T. Benjamin, et al., "Combinatorial Trigonometry with Chebyshev
  Polynomials," *Journal of Statistical Planning and Inference 14*, 2008
  (preprint: http://www.math.hmc.edu/~benjamin/papers/CombTrig.pdf, pg. 4)

"""
