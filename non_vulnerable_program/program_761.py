import f2pytest,sys
def g():
    return 'abcdefgh'
r = f2pytest.f(g)
if not r=='abcdefgh':
    sys.stderr.write('expected "abcdefgh" but got %s\\n'%r)
    sys.exit()
print 'ok'
"""
    tests.append(test)


#!/usr/bin/env python
"""

This file contains the following tests:

    Argument passing to Fortran function(<typespec>)
    Fortran function returning <typespec>
    Simple callback from Fortran
    Callback function returning <typespec>

Copyright 2000 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@ioc.ee>          
Permission to use, modify, and distribute this software is given under the
terms of the LGPL.  See http://www.fsf.org

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2000/05/01 17:10:43 $
Pearu Peterson
"""

__version__ = "$Revision: 1.12 $"[10:-1]


tests=[]
all=1 # run all tests
skip=1
#################################################################
if 0: #Template
    test={}
    test['name']=''
    test['f']="""\
      function f()
      end
"""
    test['py']="""\
