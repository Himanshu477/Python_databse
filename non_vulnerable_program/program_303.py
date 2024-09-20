from __cvs_version__ import cvs_version
cvs_minor = cvs_version[-3]
cvs_serial = cvs_version[-1]

weave_version = '%(major)d.%(minor)d.%(micro)d_%(release_level)s'\
                  '_%(cvs_minor)d.%(cvs_serial)d' % (locals ())


#!/usr/bin/env python
"""
Postpone module import to future.

Python versions: 1.5.2 - 2.3.x
Author: Pearu Peterson <pearu@cens.ioc.ee>
Created: March 2003
$Revision$
$Date$
"""
__all__ = ['ppimport','ppimport_attr']

