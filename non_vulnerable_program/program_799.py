import f2pytest,sys
r = f2pytest.f()
if r:
    sys.stderr.write('expected .false. but got %s\\n'%(r))
r = f2pytest.f2()
if not r:
    sys.stderr.write('expected .true. but got %s\\n'%(r))
    sys.exit()
print 'ok'
"""
        tests.append(test)


#!/usr/bin/env python
"""

Copyright 2000 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@ioc.ee>          
Permission to use, modify, and distribute this software is given under the
terms of the LGPL.  See http://www.fsf.org

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2001/12/17 18:11:12 $
Pearu Peterson
"""

__version__ = "$Revision: 1.12 $"[10:-1]

