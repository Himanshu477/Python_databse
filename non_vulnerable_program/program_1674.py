import os
from distutils.unixccompiler import UnixCCompiler
from numpy.distutils.exec_command import find_executable

class IntelCCompiler(UnixCCompiler):

    """ A modified Intel compiler compatible with an gcc built Python.
    """

    compiler_type = 'intel'
    cc_exe = 'icc'

    def __init__ (self, verbose=0, dry_run=0, force=0):
        UnixCCompiler.__init__ (self, verbose,dry_run, force)
        compiler = self.cc_exe
        self.set_executables(compiler=compiler,
                             compiler_so=compiler,
                             compiler_cxx=compiler,
                             linker_exe=compiler,
                             linker_so=compiler + ' -shared')

class IntelItaniumCCompiler(IntelCCompiler):
    compiler_type = 'intele'

    # On Itanium, the Intel Compiler used to be called ecc, let's search for
    # it (now it's also icc, so ecc is last in the search).
    for cc_exe in map(find_executable,['icc','ecc']):
        if cc_exe:
            break


"""
This module converts code written for Numeric to run with numpy

Makes the following changes:
 * Changes import statements (warns of use of from Numeric import *)
 * Changes import statements (using numerix) ...
 * Makes search and replace changes to:
   - .typecode()
   - .iscontiguous()
   - .byteswapped()
   - .itemsize()
   - .toscalar()
 * Converts .flat to .ravel() except for .flat = xxx or .flat[xxx]
 * Replace xxx.spacesaver() with True
 * Convert xx.savespace(?) to pass + ## xx.savespace(?)
"""
__all__ = ['fromfile', 'fromstr']

