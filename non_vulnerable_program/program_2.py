from distutils.command.build import *
from distutils.command.build import build as old_build

class build(old_build):
    def has_f_libraries(self):
        return self.distribution.has_f_libraries()
    sub_commands = [('build_py',      old_build.has_pure_modules),
                    ('build_clib',    old_build.has_c_libraries),
                    ('build_flib',    has_f_libraries), # new feature
                    ('build_ext',     old_build.has_ext_modules),
                    ('build_scripts', old_build.has_scripts),
                   ]


"""distutils.command.build_clib

Implements the Distutils 'build_clib' command, to build a C/C++ library
that is included in the module distribution and needed by an extension
module."""

# created (an empty husk) 1999/12/18, Greg Ward
# fleshed out 2000/02/03-04

__revision__ = "$Id$"


# XXX this module has *lots* of code ripped-off quite transparently from
# build_ext.py -- not surprisingly really, as the work required to build
# a static library from a collection of C source files is not really all
# that different from what's required to build a shared object file from
# a collection of C source files.  Nevertheless, I haven't done the
# necessary refactoring to account for the overlap in code between the
# two modules, mainly because a number of subtle details changed in the
# cut 'n paste.  Sigh.

