    from __cvs_version__ import cvs_version
    version_info = (major,)+cvs_version[1:]
    version = '%s.%s.%s_%s' % version_info
except:
    version = '%s_tarball' % (major)



#!/usr/bin/env python
"""

Auxiliary functions for f2py2e.

Copyright 1999,2000 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@ioc.ee>          
Permission to use, modify, and distribute this software is given under the
terms of the LGPL.  See http://www.fsf.org

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2005/07/24 19:01:55 $
Pearu Peterson
"""
__version__ = "$Revision: 1.65 $"[10:-1]

