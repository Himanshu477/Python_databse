    import numpy as N
    import os

    _path = os.path.dirname('__file__')
    lib = N.ctypeslib.load_library('code', _path)
    _typedict = {'zadd' : complex, 'sadd' : N.single,
                 'cadd' : N.csingle, 'dadd' : float}
    for name in _typedict.keys():
        val = getattr(lib, name)
        val.restype = None
        _type = _typedict[name]
        val.argtypes = [N.ctypeslib.ndpointer(_type,
                          flags='aligned, contiguous'),
                        N.ctypeslib.ndpointer(_type,
                          flags='aligned, contiguous'),
                        N.ctypeslib.ndpointer(_type,
                          flags='aligned, contiguous,'\
                                'writeable'),
                        N.ctypeslib.c_intp]

This code loads the shared library named code.{ext} located in the
same path as this file. It then adds a return type of void to the
functions contained in the library. It also adds argument checking to
the functions in the library so that ndarrays can be passed as the
first three arguments along with an integer (large enough to hold a
pointer on the platform) as the fourth argument.

Setting up the filtering function is similar and allows the filtering
function to be called with ndarray arguments as the first two
arguments and with pointers to integers (large enough to handle the
strides and shape of an ndarray) as the last two arguments.:

.. code-block:: python

    lib.dfilter2d.restype=None
    lib.dfilter2d.argtypes = [N.ctypeslib.ndpointer(float, ndim=2,
                                           flags='aligned'),
                              N.ctypeslib.ndpointer(float, ndim=2,
                                     flags='aligned, contiguous,'\
                                           'writeable'),
                              ctypes.POINTER(N.ctypeslib.c_intp),
                              ctypes.POINTER(N.ctypeslib.c_intp)]

Next, define a simple selection function that chooses which addition
function to call in the shared library based on the data-type:

.. code-block:: python

    def select(dtype):
        if dtype.char in ['?bBhHf']:
            return lib.sadd, single
        elif dtype.char in ['F']:
            return lib.cadd, csingle
        elif dtype.char in ['DG']:
            return lib.zadd, complex
        else:
            return lib.dadd, float
        return func, ntype

Finally, the two functions to be exported by the interface can be
written simply as:

.. code-block:: python

    def add(a, b):
        requires = ['CONTIGUOUS', 'ALIGNED']
        a = N.asanyarray(a)
        func, dtype = select(a.dtype)
        a = N.require(a, dtype, requires)
        b = N.require(b, dtype, requires)
        c = N.empty_like(a)
        func(a,b,c,a.size)
        return c

and:

.. code-block:: python

    def filter2d(a):
        a = N.require(a, float, ['ALIGNED'])
        b = N.zeros_like(a)
        lib.dfilter2d(a, b, a.ctypes.strides, a.ctypes.shape)
        return b


Conclusion
----------

.. index::
   single: ctypes

Using ctypes is a powerful way to connect Python with arbitrary
C-code. It's advantages for extending Python include

- clean separation of C-code from Python code

    - no need to learn a new syntax except Python and C

    - allows re-use of C-code

    - functionality in shared libraries written for other purposes can be
      obtained with a simple Python wrapper and search for the library.


- easy integration with NumPy through the ctypes attribute

- full argument checking with the ndpointer class factory

It's disadvantages include

- It is difficult to distribute an extension module made using ctypes
  because of a lack of support for building shared libraries in
  distutils (but I suspect this will change in time).

- You must have shared-libraries of your code (no static libraries).

- Very little support for C++ code and it's different library-calling
  conventions. You will probably need a C-wrapper around C++ code to use
  with ctypes (or just use Boost.Python instead).

Because of the difficulty in distributing an extension module made
using ctypes, f2py is still the easiest way to extend Python for
package creation. However, ctypes is a close second and will probably
be growing in popularity now that it is part of the Python
distribution. This should bring more features to ctypes that should
eliminate the difficulty in extending Python and distributing the
extension using ctypes.


Additional tools you may find useful
====================================

These tools have been found useful by others using Python and so are
included here. They are discussed separately because I see them as
either older ways to do things more modernly handled by f2py, weave,
Pyrex, or ctypes (SWIG, PyFort, PyInline) or because I don't know much
about them (SIP, Boost, Instant). I have not added links to these
methods because my experience is that you can find the most relevant
link faster using Google or some other search engine, and any links
provided here would be quickly dated. Do not assume that just because
it is included in this list, I don't think the package deserves your
attention. I'm including information about these packages because many
people have found them useful and I'd like to give you as many options
as possible for tackling the problem of easily integrating your code.


SWIG
----

.. index::
   single: swig

Simplified Wrapper and Interface Generator (SWIG) is an old and fairly
stable method for wrapping C/C++-libraries to a large variety of other
languages. It does not specifically understand NumPy arrays but can be
made useable with NumPy through the use of typemaps. There are some
sample typemaps in the numpy/doc/swig directory under numpy.i along
with an example module that makes use of them. SWIG excels at wrapping
large C/C++ libraries because it can (almost) parse their headers and
auto-produce an interface. Technically, you need to generate a ``.i``
file that defines the interface. Often, however, this ``.i`` file can
be parts of the header itself. The interface usually needs a bit of
tweaking to be very useful. This ability to parse C/C++ headers and
auto-generate the interface still makes SWIG a useful approach to
adding functionalilty from C/C++ into Python, despite the other
methods that have emerged that are more targeted to Python. SWIG can
actually target extensions for several languages, but the typemaps
usually have to be language-specific. Nonetheless, with modifications
to the Python-specific typemaps, SWIG can be used to interface a
library with other languages such as Perl, Tcl, and Ruby.

My experience with SWIG has been generally positive in that it is
relatively easy to use and quite powerful. I used to use it quite
often before becoming more proficient at writing C-extensions.
However, I struggled writing custom interfaces with SWIG because it
must be done using the concept of typemaps which are not Python
specific and are written in a C-like syntax. Therefore, I tend to
prefer other gluing strategies and would only attempt to use SWIG to
wrap a very-large C/C++ library. Nonetheless, there are others who use
SWIG quite happily.


SIP
---

.. index::
   single: SIP

SIP is another tool for wrapping C/C++ libraries that is Python
specific and appears to have very good support for C++. Riverbank
Computing developed SIP in order to create Python bindings to the QT
library. An interface file must be written to generate the binding,
but the interface file looks a lot like a C/C++ header file. While SIP
is not a full C++ parser, it understands quite a bit of C++ syntax as
well as its own special directives that allow modification of how the
Python binding is accomplished. It also allows the user to define
mappings between Python types and C/C++ structrues and classes.


Boost Python
------------

.. index::
   single: Boost.Python

Boost is a repository of C++ libraries and Boost.Python is one of
those libraries which provides a concise interface for binding C++
classes and functions to Python. The amazing part of the Boost.Python
approach is that it works entirely in pure C++ without introducing a
new syntax. Many users of C++ report that Boost.Python makes it
possible to combine the best of both worlds in a seamless fashion. I
have not used Boost.Python because I am not a big user of C++ and
using Boost to wrap simple C-subroutines is usually over-kill. It's
primary purpose is to make C++ classes available in Python. So, if you
have a set of C++ classes that need to be integrated cleanly into
Python, consider learning about and using Boost.Python.


Instant
-------

.. index::
   single: Instant

This is a relatively new package (called pyinstant at sourceforge)
that builds on top of SWIG to make it easy to inline C and C++ code in
Python very much like weave. However, Instant builds extension modules
on the fly with specific module names and specific method names. In
this repsect it is more more like f2py in its behavior. The extension
modules are built on-the fly (as long as the SWIG is installed). They
can then be imported. Here is an example of using Instant with NumPy
arrays (adapted from the test2 included in the Instant distribution):

.. code-block:: python

    code="""
    PyObject* add(PyObject* a_, PyObject* b_){
      /*
      various checks
      */
      PyArrayObject* a=(PyArrayObject*) a_;
      PyArrayObject* b=(PyArrayObject*) b_;
      int n = a->dimensions[0];
      int dims[1];
      dims[0] = n;
      PyArrayObject* ret;
      ret = (PyArrayObject*) PyArray_FromDims(1, dims, NPY_DOUBLE);
      int i;
      char *aj=a->data;
      char *bj=b->data;
      double *retj = (double *)ret->data;
      for (i=0; i < n; i++) {
        *retj++ = *((double *)aj) + *((double *)bj);
        aj += a->strides[0];
        bj += b->strides[0];
      }
    return (PyObject *)ret;
    }
    """
