    import numpy
    py_ext = Extension('mine', ['mine.pyx'],
             include_dirs=[numpy.get_include()])

    setup(name='mine', description='Nothing',
          ext_modules=[pyx_ext],
          cmdclass = {'build_ext':build_ext})

Adding the NumPy include directory is, of course, only necessary if
you are using NumPy arrays in the extension module (which is what I
assume you are using Pyrex for). The distutils extensions in NumPy
also include support for automatically producing the extension-module
and linking it from a ``.pyx`` file. It works so that if the user does
not have Pyrex installed, then it looks for a file with the same
file-name but a ``.c`` extension which it then uses instead of trying
to produce the ``.c`` file again.

Pyrex does not natively understand NumPy arrays. However, it is not
difficult to include information that lets Pyrex deal with them
usefully. In fact, the numpy.random.mtrand module was written using
Pyrex so an example of Pyrex usage is already included in the NumPy
source distribution. That experience led to the creation of a standard
c_numpy.pxd file that you can use to simplify interacting with NumPy
array objects in a Pyrex-written extension. The file may not be
complete (it wasn't at the time of this writing). If you have
additions you'd like to contribute, please send them. The file is
located in the .../site-packages/numpy/doc/pyrex directory where you
have Python installed. There is also an example in that directory of
using Pyrex to construct a simple extension module. It shows that
Pyrex looks a lot like Python but also contains some new syntax that
is necessary in order to get C-like speed.

If you just use Pyrex to compile a standard Python module, then you
will get a C-extension module that runs either as fast or, possibly,
more slowly than the equivalent Python module. Speed increases are
possible only when you use cdef to statically define C variables and
use a special construct to create for loops:

.. code-block:: none

    cdef int i
    for i from start <= i < stop

Let's look at two examples we've seen before to see how they might be
implemented using Pyrex. These examples were compiled into extension
modules using Pyrex-0.9.3.1.


Pyrex-add
---------

Here is part of a Pyrex-file I named add.pyx which implements the add
functions we previously implemented using f2py:

.. code-block:: none

    cimport c_numpy
