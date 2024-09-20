    from fcompiler import new_fcompiler
    compiler = new_fcompiler(compiler='compaq')
    compiler.customize()
    print compiler.get_version()


#!/usr/bin/env python
"""
cpuinfo

Copyright 2002 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@cens.ioc.ee>          
Permission to use, modify, and distribute this software is given under the 
terms of the SciPy (BSD style) license.  See LICENSE.txt that came with
this distribution for specifics.

Note:  This should be merged into proc at some point.  Perhaps proc should
be returning classes like this instead of using dictionaries.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Revision$
$Date$
Pearu Peterson
"""

__version__ = "$Id$"

__all__ = ['cpu']

