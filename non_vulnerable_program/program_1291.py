    from numpy_distutils.core import setup
    setup(name = 'f2py_example',
          description       = "F2PY Users Guide examples",
          author            = "Pearu Peterson",
          author_email      = "pearu@cens.ioc.ee",
          ext_modules = [ext1,ext2]
          )
# End of setup_example.py


#!/usr/bin/env python
"""

f2py2e - Fortran to Python C/API generator. 2nd Edition.
         See __usage__ below.

Copyright 1999--2005 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@cens.ioc.ee>          
Permission to use, modify, and distribute this software is given under the
terms of the LGPL.  See http://www.fsf.org

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2005/05/06 08:31:19 $
Pearu Peterson
"""
__version__ = "$Revision: 1.90 $"[10:-1]

