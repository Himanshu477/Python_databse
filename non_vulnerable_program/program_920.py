import sys
import re
import os

unused_internal_funcs = ['__Pyx_PrintItem',
                         '__Pyx_PrintNewline',
                         '__Pyx_ReRaise',
                         '__Pyx_GetExcValue',
                         '__Pyx_ArgTypeTest',
                         '__Pyx_TypeTest',
                         '__Pyx_SetVtable',
                         '__Pyx_GetVtable',
                         '__Pyx_CreateClass']

if __name__ == '__main__':
    os.system('pyrexc mtrand.pyx')
    mtrand_c = open('mtrand.c', 'r')
    processed = open('mtrand_pp.c', 'w')
    unused_funcs_str = '(' + '|'.join(unused_internal_funcs) + ')'
    uifpat = re.compile(r'static \w+ \*?'+unused_funcs_str+r'.*/\*proto\*/')
    for linenum, line in enumerate(mtrand_c):
        m = re.match(r'^(\s+arrayObject\w*\s*=\s*[(])[(]PyObject\s*[*][)]',
                     line)
        if m:
            line = '%s(PyArrayObject *)%s' % (m.group(1), line[m.end():])
        m = uifpat.match(line)
        if m:
            line = ''
        m = re.search(unused_funcs_str, line)
        if m:
            print >>sys.stderr, \
                "%s was declared unused, but is used at line %d" % (m.group(),
                                                                    linenum+1)
        processed.write(line)
    mtrand_c.close()
    processed.close()
    os.rename('mtrand_pp.c', 'mtrand.c')


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
$Revision: 1.1 $
$Date: 2005/04/09 19:29:34 $
Pearu Peterson
"""

__version__ = "$Id: cpuinfo.py,v 1.1 2005/04/09 19:29:34 pearu Exp $"

__all__ = ['cpu']

