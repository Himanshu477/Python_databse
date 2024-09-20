    from numpy import __version__ as numpy_version
except ImportError:
    numpy_version = 'N/A'

__all__ = ['main']

__usage__ = """
F2PY G3 --- The third generation of Fortran to Python Interface Generator
=========================================================================

Description
-----------

f2py program generates a Python C/API file (<modulename>module.c) that
contains wrappers for given Fortran functions and data so that they
can be accessed from Python. With the -c option the corresponding
extension modules are built.

Options
-------

  --3g-numpy       Use numpy.f2py.lib tool, the 3rd generation of F2PY,
                   with NumPy support.
  --2d-numpy       Use numpy.f2py tool with NumPy support. [DEFAULT]
  --2d-numeric     Use f2py2e tool with Numeric support.
  --2d-numarray    Use f2py2e tool with Numarray support.

"""

