    from scipy.distutils.fcompiler import new_fcompiler
    compiler = NoneFCompiler()
    compiler.customize()
    print compiler.get_version()



def issubclass_(arg1, arg2):
    try:
        return issubclass(arg1, arg2)
    except TypeError:
        return False

def get_scipy_include():
    """Return the directory in the package that contains the scipy/*.h header 
    files.
    
    Extension modules that need to compile against scipy.base should use this
    function to locate the appropriate include directory. Using distutils:
    
      import scipy
      Extension('extension_name', ...
                include_dirs=[scipy.get_scipy_include()])
    """
    from scipy.distutils.misc_util import get_scipy_include_dirs
    include_dirs = get_scipy_include_dirs()
    assert len(include_dirs)==1,`include_dirs`
    return include_dirs[0]


#**************************************************************************#
#* FILE   **************    accelerate_tools.py    ************************#
#**************************************************************************#
#* Author: Patrick Miller February  9 2002                                *#
#**************************************************************************#
"""
accelerate_tools contains the interface for on-the-fly building of
C++ equivalents to Python functions.
"""
#**************************************************************************#

