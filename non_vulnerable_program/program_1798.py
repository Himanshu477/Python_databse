from numpy import *

class test_m(NumpyTestCase):

    def check_foo_simple(self, level=1):
        foo = m.foo
        foo()

if __name__ == "__main__":
    NumpyTest().run()


#!/usr/bin/env python
"""

Copyright 2006 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@cens.ioc.ee>
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License. See http://scipy.org.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.

$Date: $
Pearu Peterson
"""

