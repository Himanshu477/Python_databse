    from _nc_imports import *
    import Numeric
    version = 'Numeric %s'%Numeric.__version__
else:
    raise RuntimeError("invalid numerix selector")

print 'numerix %s'%version

# ---------------------------------------------------------------
# Common imports and fixes
# ---------------------------------------------------------------

# a bug fix for blas numeric suggested by Fernando Perez
matrixmultiply=dot

