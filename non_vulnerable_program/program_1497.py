import os
import sys
from distutils.command.build import build as old_build
from distutils.util import get_platform

class build(old_build):

    sub_commands = [('config_fc',     lambda *args: 1),
                    ('build_src',     old_build.has_ext_modules),
                    ] + old_build.sub_commands

    def finalize_options(self):
        build_scripts = self.build_scripts
        old_build.finalize_options(self)
        plat_specifier = ".%s-%s" % (get_platform(), sys.version[0:3])
        if build_scripts is None:
            self.build_scripts = os.path.join(self.build_base,
                                              'scripts' + plat_specifier)




#!/usr/bin/env python
"""
Defines FortranReader classes for reading Fortran codes from
files and strings. FortranReader handles comments and line continuations
of both fix and free format Fortran codes.

Copyright 2006 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@cens.ioc.ee>
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License. See http://scipy.org.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
Pearu Peterson
"""

