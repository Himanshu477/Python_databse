from low to high, thus ``array([1,2,3])`` will be treated as the series
``T_0 + 2*T_1 + 3*T_2``

Constants
---------
- chebdomain -- Chebyshev series default domain
- chebzero -- Chebyshev series that evaluates to 0.
- chebone -- Chebyshev series that evaluates to 1.
- chebx -- Chebyshev series of the identity map (x).

Arithmetic
----------
- chebadd -- add a Chebyshev series to another.
- chebsub -- subtract a Chebyshev series from another.
- chebmul -- multiply a Chebyshev series by another
- chebdiv -- divide one Chebyshev series by another.
- chebval -- evaluate a Chebyshev series at given points.

Calculus
--------
- chebder -- differentiate a Chebyshev series.
- chebint -- integrate a Chebyshev series.

Misc Functions
--------------
- chebfromroots -- create a Chebyshev series with specified roots.
- chebroots -- find the roots of a Chebyshev series.
- chebvander -- Vandermode like matrix for Chebyshev polynomials.
- chebfit -- least squares fit returning a Chebyshev series.
- chebtrim -- trim leading coefficients from a Chebyshev series.
- chebline -- Chebyshev series of given straight line
- cheb2poly -- convert a Chebyshev series to a polynomial.
- poly2cheb -- convert a polynomial to a Chebyshev series.

Classes
-------
- Chebyshev -- Chebyshev series class.

Notes
-----
The implementations of multiplication, division, integration, and
differentiation use the algebraic identities:

.. math ::
    T_n(x) = \\frac{z^n + z^{-n}}{2} \\\\
    z\\frac{dx}{dz} = \\frac{z - z^{-1}}{2}.

where

.. math :: x = \\frac{z + z^{-1}}{2}.

These identities allow a Chebyshev series to be expressed as a finite,
symmetric Laurent series. These sorts of Laurent series are referred to as
z-series in this module.

"""
