from numpy.testing import Tester
test = Tester().test
bench = Tester().bench


"""Functions for dealing with Chebyshev series.

This module provide s a number of functions that are useful in dealing with
Chebyshev series as well as a ``Chebyshev`` class that encapsuletes the usual
arithmetic operations. All the Chebyshev series are assumed to be ordered
