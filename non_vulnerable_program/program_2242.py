fromfile() function and .tofile() method to read and write numpy arrays
directly (mind your byteorder though!) If a good C or C++ library exists that
read the data, one can wrap that library with a variety of techniques (see
xxx) though that certainly is much more work and requires significantly more
advanced knowledge to interface with C or C++.

Use of Special Libraries
------------------------

There are libraries that can be used to generate arrays for special purposes
and it isn't possible to enumerate all of them. The most common uses are use
of the many array generation functions in random that can generate arrays of
random values, and some utility functions to generate special matrices (e.g.
diagonal)

"""


"""
========
Glossary
========

along an axis
    Axes are defined for arrays with more than one dimension.  A
    2-dimensional array has two corresponding axes: the first running
    vertically downwards across rows (axis 0), and the second running
    horizontally across columns (axis 1).

    Many operation can take place along one of these axes.  For example,
    we can sum each row of an array, in which case we operate along
    columns, or axis 1::

      >>> x = np.arange(12).reshape((3,4))

      >>> x
      array([[ 0,  1,  2,  3],
             [ 4,  5,  6,  7],
             [ 8,  9, 10, 11]])

      >>> x.sum(axis=1)
      array([ 6, 22, 38])

array
    A homogeneous container of numerical elements.  Each element in the
    array occupies a fixed amount of memory (hence homogeneous), and
    can be a numerical element of a single type (such as float, int
    or complex) or a combination (such as ``(float, int, float)``).  Each
    array has an associated data-type (or ``dtype``), which describes
    the numerical type of its elements::

      >>> x = np.array([1, 2, 3], float)

      >>> x
      array([ 1.,  2.,  3.])

      >>> x.dtype # floating point number, 64 bits of memory per element
      dtype('float64')


      # More complicated data type: each array element is a combination of
      # and integer and a floating point number
      >>> np.array([(1, 2.0), (3, 4.0)], dtype=[('x', int), ('y', float)])
      array([(1, 2.0), (3, 4.0)],
            dtype=[('x', '<i4'), ('y', '<f8')])

    Fast element-wise operations, called `ufuncs`_, operate on arrays.

array_like
    Any sequence that can be interpreted as an ndarray.  This includes
    nested lists, tuples, scalars and existing arrays.

attribute
    A property of an object that can be accessed using ``obj.attribute``,
    e.g., ``shape`` is an attribute of an array::

      >>> x = np.array([1, 2, 3])
      >>> x.shape
      (3,)

BLAS
    `Basic Linear Algebra Subprograms <http://en.wikipedia.org/wiki/BLAS>`_

broadcast
    NumPy can do operations on arrays whose shapes are mismatched::

      >>> x = np.array([1, 2])
      >>> y = np.array([[3], [4]])

      >>> x
      array([1, 2])

      >>> y
      array([[3],
             [4]])

      >>> x + y
      array([[4, 5],
             [5, 6]])

    See `doc.broadcasting`_ for more information.

C order
    See `row-major`

column-major
    A way to represent items in a N-dimensional array in the 1-dimensional
    computer memory. In column-major order, the leftmost index "varies the
    fastest": for example the array::

         [[1, 2, 3],
          [4, 5, 6]]

    is represented in the column-major order as::

        [1, 4, 2, 5, 3, 6]

    Column-major order is also known as the Fortran order, as the Fortran
    programming language uses it.

decorator
    An operator that transforms a function.  For example, a ``log``
    decorator may be defined to print debugging information upon
    function execution::

      >>> def log(f):
      ...     def new_logging_func(*args, **kwargs):
      ...         print "Logging call with parameters:", args, kwargs
      ...         return f(*args, **kwargs)
      ...
      ...     return new_logging_func

    Now, when we define a function, we can "decorate" it using ``log``::

      >>> @log
      ... def add(a, b):
      ...     return a + b

    Calling ``add`` then yields:

    >>> add(1, 2)
    Logging call with parameters: (1, 2) {}
    3

dictionary
    Resembling a language dictionary, which provides a mapping between
    words and descriptions thereof, a Python dictionary is a mapping
    between two objects::

      >>> x = {1: 'one', 'two': [1, 2]}

    Here, `x` is a dictionary mapping keys to values, in this case
    the integer 1 to the string "one", and the string "two" to
    the list ``[1, 2]``.  The values may be accessed using their
    corresponding keys::

      >>> x[1]
      'one'

      >>> x['two']
      [1, 2]

    Note that dictionaries are not stored in any specific order.  Also,
    most mutable (see *immutable* below) objects, such as lists, may not
    be used as keys.

    For more information on dictionaries, read the
    `Python tutorial <http://docs.python.org/tut>`_.

Fortran order
    See `column-major`

flattened
    Collapsed to a one-dimensional array. See `ndarray.flatten`_ for details.

immutable
    An object that cannot be modified after execution is called
    immutable.  Two common examples are strings and tuples.

instance
    A class definition gives the blueprint for constructing an object::

      >>> class House(object):
      ...     wall_colour = 'white'

    Yet, we have to *build* a house before it exists::

      >>> h = House() # build a house

    Now, ``h`` is called a ``House`` instance.  An instance is therefore
    a specific realisation of a class.

iterable
    A sequence that allows "walking" (iterating) over items, typically
    using a loop such as::

      >>> x = [1, 2, 3]
      >>> [item**2 for item in x]
      [1, 4, 9]

    It is often used in combintion with ``enumerate``::

      >>> for n, k in enumerate(keys):
      ...     print "Key %d: %s" % (n, k)
      ...
      Key 0: a
      Key 1: b
      Key 2: c

list
    A Python container that can hold any number of objects or items.
    The items do not have to be of the same type, and can even be
    lists themselves::

      >>> x = [2, 2.0, "two", [2, 2.0]]

    The list `x` contains 4 items, each which can be accessed individually::

      >>> x[2] # the string 'two'
      'two'

      >>> x[3] # a list, containing an integer 2 and a float 2.0
      [2, 2.0]

    It is also possible to select more than one item at a time,
    using *slicing*::

      >>> x[0:2] # or, equivalently, x[:2]
      [2, 2.0]

    In code, arrays are often conveniently expressed as nested lists::


      >>> np.array([[1, 2], [3, 4]])
      array([[1, 2],
             [3, 4]])

    For more information, read the section on lists in the `Python
    tutorial <http://docs.python.org/tut>`_.  For a mapping
    type (key-value), see *dictionary*.

mask
    A boolean array, used to select only certain elements for an operation::

      >>> x = np.arange(5)
      >>> x
      array([0, 1, 2, 3, 4])

      >>> mask = (x > 2)
      >>> mask
      array([False, False, False, True,  True], dtype=bool)

      >>> x[mask] = -1
      >>> x
      array([ 0,  1,  2,  -1, -1])

masked array
    Array that suppressed values indicated by a mask::

      >>> x = np.ma.masked_array([np.nan, 2, np.nan], [True, False, True])
      >>> x
      masked_array(data = [-- 2.0 --],
            mask = [ True False  True],
            fill_value=1e+20)

      >>> x + [1, 2, 3]
      masked_array(data = [-- 4.0 --],
            mask = [ True False  True],
            fill_value=1e+20)

    Masked arrays are often used when operating on arrays containing
    missing or invalid entries.

matrix
    A 2-dimensional ndarray that preserves its two-dimensional nature
    throughout operations.  It has certain special operations, such as ``*``
    (matrix multiplication) and ``**`` (matrix power), defined::

      >>> x = np.mat([[1, 2], [3, 4]])

      >>> x
      matrix([[1, 2],
              [3, 4]])

      >>> x**2
      matrix([[ 7, 10],
            [15, 22]])

method
    A function associated with an object.  For example, each ndarray has a
    method called ``repeat``::

      >>> x = np.array([1, 2, 3])

      >>> x.repeat(2)
      array([1, 1, 2, 2, 3, 3])

ndarray
    See *array*.

reference
    If ``a`` is a reference to ``b``, then ``(a is b) == True``.  Therefore,
    ``a`` and ``b`` are different names for the same Python object.

row-major
    A way to represent items in a N-dimensional array in the 1-dimensional
    computer memory. In row-major order, the rightmost index "varies
    the fastest": for example the array::

         [[1, 2, 3],
          [4, 5, 6]]

    is represented in the row-major order as::

        [1, 2, 3, 4, 5, 6]

    Row-major order is also known as the C order, as the C programming
    language uses it. New Numpy arrays are by default in row-major order.

self
    Often seen in method signatures, ``self`` refers to the instance
    of the associated class.  For example:

      >>> class Paintbrush(object):
      ...     color = 'blue'
      ...
      ...     def paint(self):
      ...         print "Painting the city %s!" % self.color
      ...
      >>> p = Paintbrush()
      >>> p.color = 'red'
      >>> p.paint() # self refers to 'p'
      Painting the city red!

slice
    Used to select only certain elements from a sequence::

      >>> x = range(5)
      >>> x
      [0, 1, 2, 3, 4]

      >>> x[1:3] # slice from 1 to 3 (excluding 3 itself)
      [1, 2]

      >>> x[1:5:2] # slice from 1 to 5, but skipping every second element
      [1, 3]

      >>> x[::-1] # slice a sequence in reverse
      [4, 3, 2, 1, 0]

    Arrays may have more than one dimension, each which can be sliced
    individually::

      >>> x = np.array([[1, 2], [3, 4]])
      >>> x
      array([[1, 2],
             [3, 4]])

      >>> x[:, 1]
      array([2, 4])

tuple
    A sequence that may contain a variable number of types of any
    kind.  A tuple is immutable, i.e., once constructed it cannot be
    changed.  Similar to a list, it can be indexed and sliced::

      >>> x = (1, 'one', [1, 2])

      >>> x
      (1, 'one', [1, 2])

      >>> x[0]
      1

      >>> x[:2]
      (1, 'one')

    A useful concept is "tuple unpacking", which allows variables to
    be assigned to the contents of a tuple::

      >>> x, y = (1, 2)
      >>> x, y = 1, 2

    This is often used when a function returns multiple values:

      >>> def return_many():
      ...     return 1, 'alpha'

      >>> a, b, c = return_many()
      >>> a, b, c
      (1, 'alpha', None)

      >>> a
      1
      >>> b
      'alpha'

ufunc
    Universal function.  A fast element-wise array operation.  Examples include
    ``add``, ``sin`` and ``logical_or``.

view
    An array that does not own its data, but refers to another array's
    data instead.  For example, we may create a view that only shows
    every second element of another array::

      >>> x = np.arange(5)
      >>> x
      array([0, 1, 2, 3, 4])

      >>> y = x[::2]
      >>> y
      array([0, 2, 4])

      >>> x[0] = 3 # changing x changes y as well, since y is a view on x
      >>> y
      array([3, 2, 4])

wrapper
    Python is a high-level (highly abstracted, or English-like) language.
    This abstraction comes at a price in execution speed, and sometimes
    it becomes necessary to use lower level languages to do fast
    computations.  A wrapper is code that provides a bridge between
    high and the low level languages, allowing, e.g., Python to execute
    code written in C or Fortran.

    Examples include ctypes, SWIG and Cython (which wraps C and C++)
    and f2py (which wraps Fortran).

"""


"""
=============
Miscellaneous
=============

IEEE 754 Floating Point Special Values:
-----------------------------------------------

Special values defined in numpy: nan, inf,

NaNs can be used as a poor-man's mask (if you don't care what the
original value was)

Note: cannot use equality to test NaNs. E.g.: ::

 >>> np.where(myarr == np.nan)
 >>> nan == nan  # is always False! Use special numpy functions instead.

 >>> np.nan == np.nan
 False
 >>> myarr = np.array([1., 0., np.nan, 3.])
 >>> myarr[myarr == np.nan] = 0. # doesn't work
 >>> myarr
 array([  1.,   0.,  NaN,   3.])
 >>> myarr[np.isnan(myarr)] = 0. # use this instead find
 >>> myarr
 array([ 1.,  0.,  0.,  3.])

Other related special value functions: ::

 isinf():    True if value is inf
 isfinite(): True if not nan or inf
 nan_to_num(): Map nan to 0, inf to max float, -inf to min float

The following corresponds to the usual functions except that nans are excluded from
the results: ::

 nansum()
 nanmax()
 nanmin()
 nanargmax()
 nanargmin()

 >>> x = np.arange(10.)
 >>> x[3] = np.nan
 >>> x.sum()
 nan
 >>> np.nansum(x)
 42.0

How numpy handles numerical exceptions

Default is to "warn"
But this can be changed, and it can be set individually for different kinds
of exceptions. The different behaviors are: ::

 'ignore' : ignore completely
 'warn'   : print a warning (once only)
 'raise'  : raise an exception
 'call'   : call a user-supplied function (set using seterrcall())

These behaviors can be set for all kinds of errors or specific ones: ::

 all:       apply to all numeric exceptions
 invalid:   when NaNs are generated
 divide:    divide by zero (for integers as well!)
 overflow:  floating point overflows
 underflow: floating point underflows

Note that integer divide-by-zero is handled by the same machinery.
These behaviors are set on a per-thead basis.

Examples:
------------

::

 >>> oldsettings = np.seterr(all='warn')
 >>> np.zeros(5,dtype=np.float32)/0.
 invalid value encountered in divide
 >>> j = np.seterr(under='ignore')
 >>> np.array([1.e-100])**10
 >>> j = np.seterr(invalid='raise')
 >>> np.sqrt(np.array([-1.]))
 FloatingPointError: invalid value encountered in sqrt
 >>> def errorhandler(errstr, errflag):
 ...      print "saw stupid error!"
 >>> np.seterrcall(errorhandler)
 >>> j = np.seterr(all='call')
 >>> np.zeros(5, dtype=np.int32)/0
 FloatingPointError: invalid value encountered in divide
 saw stupid error!
 >>> j = np.seterr(**oldsettings) # restore previous
                                  # error-handling settings

Interfacing to C:
-----------------
Only a survey the choices. Little detail on how each works.

1) Bare metal, wrap your own C-code manually.

 - Plusses:

   - Efficient
   - No dependencies on other tools

 - Minuses:

   - Lots of learning overhead:

     - need to learn basics of Python C API
     - need to learn basics of numpy C API
     - need to learn how to handle reference counting and love it.

   - Reference counting often difficult to get right.

     - getting it wrong leads to memory leaks, and worse, segfaults

   - API will change for Python 3.0!

2) pyrex

 - Plusses:

   - avoid learning C API's
   - no dealing with reference counting
   - can code in psuedo python and generate C code
   - can also interface to existing C code
   - should shield you from changes to Python C api
   - become pretty popular within Python community

 - Minuses:

   - Can write code in non-standard form which may become obsolete
   - Not as flexible as manual wrapping
   - Maintainers not easily adaptable to new features

Thus:

3) cython - fork of pyrex to allow needed features for SAGE

  - being considered as the standard scipy/numpy wrapping tool
  - fast indexing support for arrays

4) ctypes

 - Plusses:

   - part of Python standard library
   - good for interfacing to existing sharable libraries, particularly
     Windows DLLs
   - avoids API/reference counting issues
   - good numpy support: arrays have all these in their ctypes
     attribute: ::

       a.ctypes.data              a.ctypes.get_strides
       a.ctypes.data_as           a.ctypes.shape
       a.ctypes.get_as_parameter  a.ctypes.shape_as
       a.ctypes.get_data          a.ctypes.strides
       a.ctypes.get_shape         a.ctypes.strides_as

 - Minuses:

   - can't use for writing code to be turned into C extensions, only a wrapper tool.

5) SWIG (automatic wrapper generator)

 - Plusses:

   - around a long time
   - multiple scripting language support
   - C++ support
   - Good for wrapping large (many functions) existing C libraries

 - Minuses:

   - generates lots of code between Python and the C code

     - can cause performance problems that are nearly impossible to optimize out

   - interface files can be hard to write
   - doesn't necessarily avoid reference counting issues or needing to know API's

7) Weave

 - Plusses:

   - Phenomenal tool
   - can turn many numpy expressions into C code
   - dynamic compiling and loading of generated C code
   - can embed pure C code in Python module and have weave extract, generate interfaces
     and compile, etc.

 - Minuses:

   - Future uncertain--lacks a champion

8) Psyco

 - Plusses:

   - Turns pure python into efficient machine code through jit-like optimizations
   - very fast when it optimizes well

 - Minuses:

   - Only on intel (windows?)
   - Doesn't do much for numpy?

Interfacing to Fortran:
-----------------------
Fortran: Clear choice is f2py. (Pyfort is an older alternative, but not supported
any longer)

Interfacing to C++:
-------------------
1) CXX
2) Boost.python
3) SWIG
4) Sage has used cython to wrap C++ (not pretty, but it can be done)
5) SIP (used mainly in PyQT)

"""


