import numpy

# Define a pyrex-based extension module, using the generated sources if pyrex
# is not available.
if has_pyrex:
    pyx_sources = ['numpyx.pyx']
    cmdclass    = {'build_ext': build_ext}
else:
    pyx_sources = ['numpyx.c']
    cmdclass    = {}


pyx_ext = Extension('numpyx',
                 pyx_sources,
                 include_dirs = [numpy.get_include()])

# Call the routine which does the real work
setup(name        = 'numpyx',
      description = 'Small example on using Pyrex to write a Numpy extension',
      url         = 'http://www.scipy.org/Cookbook/Pyrex_and_NumPy',
      ext_modules = [pyx_ext],
      cmdclass    = cmdclass,
      )


# -*- coding: utf-8 -*-
"""
    linkcode
    ~~~~~~~~

    Add external links to module code in Python object descriptions.

    :copyright: Copyright 2007-2011 by the Sphinx team, see AUTHORS.
    :license: BSD, see LICENSE for details.

"""
