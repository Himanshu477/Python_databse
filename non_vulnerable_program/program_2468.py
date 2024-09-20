    import nose
    nose.runmodule()


"""
Objects for dealing with Hermite series.

This module provides a number of objects (mostly functions) useful for
dealing with Hermite series, including a `Hermite` class that
encapsulates the usual arithmetic operations.  (General information
on how this module represents and works with such polynomials is in the
docstring for its "parent" sub-package, `numpy.polynomial`).

Constants
---------
- `hermdomain` -- Hermite series default domain, [-1,1].
- `hermzero` -- Hermite series that evaluates identically to 0.
- `hermone` -- Hermite series that evaluates identically to 1.
- `hermx` -- Hermite series for the identity map, ``f(x) = x``.

Arithmetic
----------
- `hermmulx` -- multiply a Hermite series in ``P_i(x)`` by ``x``.
- `hermadd` -- add two Hermite series.
- `hermsub` -- subtract one Hermite series from another.
- `hermmul` -- multiply two Hermite series.
- `hermdiv` -- divide one Hermite series by another.
- `hermval` -- evaluate a Hermite series at given points.

Calculus
--------
- `hermder` -- differentiate a Hermite series.
- `hermint` -- integrate a Hermite series.

Misc Functions
--------------
- `hermfromroots` -- create a Hermite series with specified roots.
- `hermroots` -- find the roots of a Hermite series.
- `hermvander` -- Vandermonde-like matrix for Hermite polynomials.
- `hermfit` -- least-squares fit returning a Hermite series.
- `hermtrim` -- trim leading coefficients from a Hermite series.
- `hermline` -- Hermite series of given straight line.
- `herm2poly` -- convert a Hermite series to a polynomial.
- `poly2herm` -- convert a polynomial to a Hermite series.

Classes
-------
- `Hermite` -- A Hermite series class.

See also
--------
`numpy.polynomial`

"""
