import scipy_distutils.ccompiler

# NT stuff
# 1. Make sure libpython<version>.a exists for gcc.  If not, build it.
# 2. Force windows to use gcc (we're struggling with MSVC and g77 support) 
# 3. Force windows to use g77

# 1.  Build libpython<version> from .lib and .dll if they don't exist.    
