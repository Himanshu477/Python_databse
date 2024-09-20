import doctest
def test_suite(level=1):
    return doctest.DocTestSuite()

if __name__ == "__main__":
    ScipyTest().run()


#!/usr/bin/env python
"""
setup.py for installing F2PY

Usage:
   python setup.py install

Copyright 2001-2005 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@cens.ioc.ee>          
Permission to use, modify, and distribute this software is given under the
terms of the LGPL.  See http://www.fsf.org

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Revision: 1.32 $
$Date: 2005/01/30 17:22:14 $
Pearu Peterson
"""

__version__ = "$Id: setup.py,v 1.32 2005/01/30 17:22:14 pearu Exp $"

