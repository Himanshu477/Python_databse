from numpy import *

class test_m(NumpyTestCase):

    def check_foo_simple(self, level=1):
        foo = m.foo
        r = foo(2)
        assert isinstance(r,int32),`type(r)`
        assert_equal(r,3)

    def check_foo2_simple(self, level=1):
        foo2 = m.foo2
        r = foo2(2)
        assert isinstance(r,int32),`type(r)`
        assert_equal(r,4)

if __name__ == "__main__":
    NumpyTest().run()


#!/usr/bin/env python
"""
Tests for intent(in) arguments in subroutine-wrapped Fortran functions.

-----
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License. See http://scipy.org.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
Author: Pearu Peterson <pearu@cens.ioc.ee>
Created: Oct 2006
-----
"""

