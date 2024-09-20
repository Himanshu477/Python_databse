from ext.autosummary_generate import get_documented

CUR_DIR = os.path.dirname(__file__)
SOURCE_DIR = os.path.join(CUR_DIR, 'source', 'reference')

SKIP_LIST = """
# --- aliases:
alltrue sometrue bitwise_not cumproduct
row_stack column_stack product rank

# -- skipped:
core lib f2py dual doc emath ma rec char distutils oldnumeric numarray
testing version matlib

add_docstring add_newdoc add_newdocs fastCopyAndTranspose pkgload
conjugate disp

int0 object0 unicode0 uint0 string_ string0 void0

flagsobj

setup setupscons PackageLoader

lib.scimath.arccos lib.scimath.arcsin lib.scimath.arccosh lib.scimath.arcsinh
lib.scimath.arctanh lib.scimath.log lib.scimath.log2 lib.scimath.log10
lib.scimath.logn lib.scimath.power lib.scimath.sqrt

# --- numpy.random:
random random.info random.mtrand random.ranf random.sample random.random

# --- numpy.fft:
fft fft.Tester fft.bench fft.fftpack fft.fftpack_lite fft.helper
fft.refft fft.refft2 fft.refftn fft.irefft fft.irefft2 fft.irefftn
fft.info fft.test

# --- numpy.linalg:
linalg linalg.Tester
linalg.bench linalg.info linalg.lapack_lite linalg.linalg linalg.test

# --- numpy.ctypeslib:
ctypeslib ctypeslib.test

""".split()

def main():
    p = optparse.OptionParser(__doc__)
    options, args = p.parse_args()

    if len(args) != 0:
        p.error('Wrong number of arguments')

    # prepare
    fn = os.path.join(CUR_DIR, 'dump.xml')
    if os.path.isfile(fn):
        import_phantom_module(fn)
    
    # check
    documented, undocumented = check_numpy()
    
    # report
    in_sections = {}
    for name, locations in documented.iteritems():
        for (filename, section, keyword, toctree) in locations:
            in_sections.setdefault((filename, section, keyword), []).append(name)

    print "Documented"
    print "==========\n"

    last_filename = None
    for (filename, section, keyword), names in sorted(in_sections.items()):
        if filename != last_filename:
            print "--- %s\n" % filename
        last_filename = filename
        print " ** ", section
        print format_in_columns(sorted(names))
        print "\n"

    print ""
    print "Undocumented"
    print "============\n"
    print format_in_columns(sorted(undocumented.keys()))

def check_numpy():
    documented = get_documented(glob.glob(SOURCE_DIR + '/*.rst'))
    undocumented = {}

    import numpy, numpy.fft, numpy.linalg, numpy.random
    for mod in [numpy, numpy.fft, numpy.linalg, numpy.random,
                numpy.ctypeslib, numpy.emath, numpy.ma]:
        undocumented.update(get_undocumented(documented, mod, skip=SKIP_LIST))

    for d in (documented, undocumented):
        for k in d.keys():
            if k.startswith('numpy.'):
                d[k[6:]] = d[k]
                del d[k]
    
    return documented, undocumented

def get_undocumented(documented, module, module_name=None, skip=[]):
    """
    Find out which items in Numpy are not documented.

    Returns
    -------
    undocumented : dict of bool
        Dictionary containing True for each documented item name
        and False for each undocumented one.

    """
    undocumented = {}
    
    if module_name is None:
        module_name = module.__name__
    
    for name in dir(module):
        obj = getattr(module, name)
        if name.startswith('_'): continue
        
        full_name = '.'.join([module_name, name])

        if full_name in skip: continue
        if full_name.startswith('numpy.') and full_name[6:] in skip: continue
        if not (inspect.ismodule(obj) or callable(obj) or inspect.isclass(obj)):
            continue
        
        if full_name not in documented:
            undocumented[full_name] = True
    
    return undocumented

def format_in_columns(lst):
    """
    Format a list containing strings to a string containing the items
    in columns.
    """
    lst = map(str, lst)
    col_len = max(map(len, lst)) + 2
    ncols = 80//col_len
    if ncols == 0:
        ncols = 1

    if len(lst) % ncols == 0:
        nrows = len(lst)//ncols
    else:
        nrows = 1 + len(lst)//ncols
    
    fmt = ' %%-%ds ' % (col_len-2)
    
    lines = []
    for n in range(nrows):
        lines.append("".join([fmt % x for x in lst[n::nrows]]))
    return "\n".join(lines)

if __name__ == "__main__": main()


"""
============
Array basics
============

Array types and conversions between types
=========================================

Numpy supports a much greater variety of numerical types than Python does.
This section shows which are available, and how to modify an array's data-type.

==========  =========================================================
Data type   Description
==========  =========================================================
bool        Boolean (True or False) stored as a byte
int         Platform integer (normally either ``int32`` or ``int64``)
int8        Byte (-128 to 127)
int16       Integer (-32768 to 32767)
int32       Integer (-2147483648 to 2147483647)
int64       Integer (9223372036854775808 to 9223372036854775807)
uint8       Unsigned integer (0 to 255)
uint16      Unsigned integer (0 to 65535)
uint32      Unsigned integer (0 to 4294967295)
uint64      Unsigned integer (0 to 18446744073709551615)
float       Shorthand for ``float64``.
float32     Single precision float: sign bit, 8 bits exponent,
            23 bits mantissa
float64     Double precision float: sign bit, 11 bits exponent,
            52 bits mantissa
complex     Shorthand for ``complex128``.
complex64   Complex number, represented by two 32-bit floats (real
            and imaginary components)
complex128  Complex number, represented by two 64-bit floats (real
            and imaginary components)
==========  =========================================================

Numpy numerical types are instances of ``dtype`` (data-type) objects, each
having unique characteristics.  Once you have imported NumPy using

  ::

    >>> import numpy as np

the dtypes are available as ``np.bool``, ``np.float32``, etc.

Advanced types, not listed in the table above, are explored in
section `link_here`.

There are 5 basic numerical types representing booleans (bool), integers (int),
unsigned integers (uint) floating point (float) and complex. Those with numbers
in their name indicate the bitsize of the type (i.e. how many bits are needed
to represent a single value in memory).  Some types, such as ``int`` and
``intp``, have differing bitsizes, dependent on the platforms (e.g. 32-bit
vs. 64-bit machines).  This should be taken into account when interfacing
with low-level code (such as C or Fortran) where the raw memory is addressed.

Data-types can be used as functions to convert python numbers to array scalars
(see the array scalar section for an explanation), python sequences of numbers
to arrays of that type, or as arguments to the dtype keyword that many numpy
functions or methods accept. Some examples::

    >>> import numpy as np
    >>> x = np.float32(1.0)
    >>> x
    1.0
    >>> y = np.int_([1,2,4])
    >>> y
    array([1, 2, 4])
    >>> z = np.arange(3, dtype=np.uint8)
    array([0, 1, 2], dtype=uint8)

Array types can also be referred to by character codes, mostly to retain
backward compatibility with older packages such as Numeric.  Some
documentation may still refer to these, for example::

  >>> np.array([1, 2, 3], dtype='f')
  array([ 1.,  2.,  3.], dtype=float32)

We recommend using dtype objects instead.

To convert the type of an array, use the .astype() method (preferred) or
the type itself as a function. For example: ::

    >>> z.astype(float)
    array([0.,  1.,  2.])
    >>> np.int8(z)
    array([0, 1, 2], dtype=int8)

Note that, above, we use the *Python* float object as a dtype.  NumPy knows
that ``int`` refers to ``np.int``, ``bool`` means ``np.bool`` and
that ``float`` is ``np.float``.  The other data-types do not have Python
equivalents.

To determine the type of an array, look at the dtype attribute::

    >>> z.dtype
    dtype('uint8')

dtype objects also contain information about the type, such as its bit-width
and its byte-order. See xxx for details.  The data type can also be used
indirectly to query properties of the type, such as whether it is an integer::

    >>> d = np.dtype(int)
    >>> d
    dtype('int32')

    >>> np.issubdtype(d, int)
    True

    >>> np.issubdtype(d, float)
    False


Array Scalars
=============

Numpy generally returns elements of arrays as array scalars (a scalar
with an associated dtype).  Array scalars differ from Python scalars, but
for the most part they can be used interchangeably (the primary
exception is for versions of Python older than v2.x, where integer array
scalars cannot act as indices for lists and tuples).  There are some
exceptions, such as when code requires very specific attributes of a scalar
or when it checks specifically whether a value is a Python scalar. Generally,
problems are easily fixed by explicitly converting array scalars
to Python scalars, using the corresponding Python type function
(e.g., ``int``, ``float``, ``complex``, ``str``, ``unicode``).

The primary advantage of using array scalars is that
they preserve the array type (Python may not have a matching scalar type
available, e.g. ``int16``).  Therefore, the use of array scalars ensures
identical behaviour between arrays and scalars, irrespective of whether the
value is inside an array or not.  NumPy scalars also have many of the same
methods arrays do.

"""


"""
==============
Array Creation
==============

Introduction
============

There are 5 general mechanisms for creating arrays:

1) Conversion from other Python structures (e.g., lists, tuples)
2) Intrinsic numpy array array creation objects (e.g., arange, ones, zeros, etc.)
3) Reading arrays from disk, either from standard or custom formats
4) Creating arrays from raw bytes through the use of strings or buffers
5) Use of special library functions (e.g., random)

This section will not cover means of replicating, joining, or otherwise
expanding or mutating existing arrays. Nor will it cover creating object
arrays or record arrays. Both of those are covered in their own sections.

Converting Python array_like Objects to Numpy Arrays
====================================================

In general, numerical data arranged in an array-like structure in Python can
be converted to arrays through the use of the array() function. The most obvious
examples are lists and tuples. See the documentation for array() for details for
its use. Some objects may support the array-protocol and allow conversion to arrays
this way. A simple way to find out if the object can be converted to a numpy array
using array() is simply to try it interactively and see if it works! (The Python Way).

Examples: ::

 >>> x = np.array([2,3,1,0])
 >>> x = np.array([2, 3, 1, 0])
 >>> x = np.array([[1,2.0],[0,0],(1+1j,3.)]) # note mix of tuple and lists, and types
 >>> x = np.array([[ 1.+0.j, 2.+0.j], [ 0.+0.j, 0.+0.j], [ 1.+1.j, 3.+0.j]])

Intrinsic Numpy Array Creation
==============================

Numpy has built-in functions for creating arrays from scratch:

zeros(shape) will create an array filled with 0 values with the specified
shape. The default dtype is float64.

``>>> np.zeros((2, 3))
array([[ 0., 0., 0.], [ 0., 0., 0.]])``

ones(shape) will create an array filled with 1 values. It is identical to
zeros in all other respects.

arange() will create arrays with regularly incrementing values. Check the
docstring for complete information on the various ways it can be used. A few
examples will be given here: ::

 >>> np.arange(10)
 array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
 >>> np.arange(2, 10, dtype=np.float)
 array([ 2., 3., 4., 5., 6., 7., 8., 9.])
 >>> np.arange(2, 3, 0.1)
 array([ 2. , 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9])

Note that there are some subtleties regarding the last usage that the user
should be aware of that are described in the arange docstring.

linspace() will create arrays with a specified number of elements, and
spaced equally between the specified beginning and end values. For
example: ::

 >>> np.linspace(1., 4., 6)
 array([ 1. ,  1.6,  2.2,  2.8,  3.4,  4. ])

The advantage of this creation function is that one can guarantee the
number of elements and the starting and end point, which arange()
generally will not do for arbitrary start, stop, and step values.

indices() will create a set of arrays (stacked as a one-higher dimensioned
array), one per dimension with each representing variation in that dimension.
An examples illustrates much better than a verbal description: ::

 >>> np.indices((3,3))
 array([[[0, 0, 0], [1, 1, 1], [2, 2, 2]], [[0, 1, 2], [0, 1, 2], [0, 1, 2]]])

This is particularly useful for evaluating functions of multiple dimensions on
a regular grid.

Reading Arrays From Disk
========================

This is presumably the most common case of large array creation. The details,
of course, depend greatly on the format of data on disk and so this section
can only give general pointers on how to handle various formats.

Standard Binary Formats
-----------------------

Various fields have standard formats for array data. The following lists the
ones with known python libraries to read them and return numpy arrays (there
may be others for which it is possible to read and convert to numpy arrays so
check the last section as well)
::

 HDF5: PyTables
 FITS: PyFITS
 Others? xxx

Examples of formats that cannot be read directly but for which it is not hard
to convert are libraries like PIL (able to read and write many image formats
such as jpg, png, etc).

Common ASCII Formats
------------------------

Comma Separated Value files (CSV) are widely used (and an export and import
option for programs like Excel). There are a number of ways of reading these
files in Python. There are CSV functions in Python and functions in pylab
(part of matplotlib).

More generic ascii files can be read using the io package in scipy.

Custom Binary Formats
---------------------

There are a variety of approaches one can use. If the file has a relatively
simple format then one can write a simple I/O library and use the numpy
