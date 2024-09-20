import scipy.distutils.ccompiler

# NT stuff
# 1. Make sure libpython<version>.a exists for gcc.  If not, build it.
# 2. Force windows to use gcc (we're struggling with MSVC and g77 support) 
#    --> this is done in scipy/distutils/ccompiler.py
# 3. Force windows to use g77

