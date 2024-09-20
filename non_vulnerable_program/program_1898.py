import numpy

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# _Vector extension module
_Vector = Extension("_Vector",
                    ["Vector_wrap.cxx",
                     "Vector.cxx"],
                    include_dirs = [numpy_include],
                    )

# _Matrix extension module
_Matrix = Extension("_Matrix",
                    ["Matrix_wrap.cxx",
                     "Matrix.cxx"],
                    include_dirs = [numpy_include],
                    )

# _Tensor extension module
_Tensor = Extension("_Tensor",
                    ["Tensor_wrap.cxx",
                     "Tensor.cxx"],
                    include_dirs = [numpy_include],
                    )

# NumyTypemapTests setup
setup(name        = "NumpyTypemapTests",
      description = "Functions that work on arrays",
      author      = "Bill Spotz",
      py_modules  = ["Vector", "Matrix", "Tensor"],
      ext_modules = [_Vector , _Matrix , _Tensor ]
      )


#! /usr/bin/env python

# System imports
