from numpy import *

class test_m(NumpyTestCase):

    def check_foo_simple(self, level=1):
        a = m.myt(2)
        assert_equal(a.flag,2)
        assert isinstance(a,m.myt),`a`
        r = m.foo(a)
        assert isinstance(r,m.myt),`r`
        assert r is a
        assert_equal(r.flag,3)
        assert_equal(a.flag,3)

        a.flag = 5
        assert_equal(r.flag,5)

        #s = m.foo((5,))

    def check_foo2_simple(self, level=1):
        a = m.myt(2)
        assert_equal(a.flag,2)
        assert isinstance(a,m.myt),`a`
        r = m.foo2(a)
        assert isinstance(r,m.myt),`r`
        assert r is not a
        assert_equal(a.flag,2)
        assert_equal(r.flag,4)


if __name__ == "__main__":
    NumpyTest().run()


#!/usr/bin/env python
"""
Tests for module with scalar derived types and subprograms.

-----
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License. See http://scipy.org.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
Author: Pearu Peterson <pearu@cens.ioc.ee>
Created: Oct 2006
-----
"""

