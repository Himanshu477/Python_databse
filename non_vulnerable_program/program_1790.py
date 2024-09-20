from numpy import *

class test_m(NumpyTestCase):

    def check_foo_simple(self, level=1):
        a = m.myt(2)
        assert_equal(a.flag,2)
        assert isinstance(a,m.myt),`a`
        r = m.foo(a)
        assert isinstance(r,m.myt),`r`
        assert_equal(r.flag,3)
        assert_equal(a.flag,2)
        
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

