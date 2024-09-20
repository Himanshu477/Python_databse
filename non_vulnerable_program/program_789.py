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

Test arguments against all combinations of intent attributes.

Copyright 2000 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@ioc.ee>          
Permission to use, modify, and distribute this software is given under the
terms of the LGPL.  See http://www.fsf.org

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2000/09/17 16:10:27 $
Pearu Peterson
"""

__version__ = "$Revision: 1.3 $"[10:-1]

