import os
import sys
import inspect
import types
from numpy.core.numerictypes import obj2sctype, generic
from numpy.core.multiarray import dtype as _dtype
from numpy.core import product, ndarray

__all__ = ['issubclass_', 'get_numpy_include', 'issubsctype',
           'issubdtype', 'deprecate', 'get_numarray_include',
           'get_include', 'info', 'source', 'who',
           'byte_bounds', 'may_share_memory']

def issubclass_(arg1, arg2):
    try:
        return issubclass(arg1, arg2)
    except TypeError:
        return False

def issubsctype(arg1, arg2):
    return issubclass(obj2sctype(arg1), obj2sctype(arg2))

def issubdtype(arg1, arg2):
    if issubclass_(arg2, generic):
        return issubclass(_dtype(arg1).type, arg2)
    mro = _dtype(arg2).type.mro()
    if len(mro) > 1:
        val = mro[1]
    else:
        val = mro[0]
    return issubclass(_dtype(arg1).type, val)

def get_include():
    """Return the directory in the package that contains the numpy/*.h header
    files.

    Extension modules that need to compile against numpy should use this
    function to locate the appropriate include directory. Using distutils:

      import numpy
      Extension('extension_name', ...
                include_dirs=[numpy.get_include()])
    """
    import numpy
    if numpy.show_config is None:
        # running from numpy source directory
        d = os.path.join(os.path.dirname(numpy.__file__), 'core', 'include')
    else:
        # using installed numpy core headers
        import numpy.core as core
        d = os.path.join(os.path.dirname(core.__file__), 'include')
    return d

def get_numarray_include(type=None):
    """Return the directory in the package that contains the numpy/*.h header
    files.

    Extension modules that need to compile against numpy should use this
    function to locate the appropriate include directory. Using distutils:

      import numpy
      Extension('extension_name', ...
                include_dirs=[numpy.get_numarray_include()])
    """
    from numpy.numarray import get_numarray_include_dirs
    include_dirs = get_numarray_include_dirs()
    if type is None:
        return include_dirs[0]
    else:
        return include_dirs + [get_include()]


if sys.version_info < (2, 4):
    # Can't set __name__ in 2.3
