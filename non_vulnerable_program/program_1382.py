from cPickle import dump, dumps

# functions that are now methods

def take(a, indices, axis=0):
    a = asarray(a)
    return a.take(indices, axis)

def reshape(a, newshape):
    """Change the shape of a to newshape.  Return a new view object.
    """
    return asarray(a).reshape(newshape)

def choose(a, choices):
    a = asarray(a)
    return a.choose(choices)

def repeat(a, repeats, axis=0):
    """repeat elements of a repeats times along axis
       repeats is a sequence of length a.shape[axis]
       telling how many times to repeat each element.
       If repeats is an integer, it is interpreted as
       a tuple of length a.shape[axis] containing repeats.
       The argument a can be anything array(a) will accept.
    """
    a = array(a, copy=False)
    return a.repeat(repeats, axis)

def put (a, ind, v):
    """put(a, ind, v) results in a[n] = v[n] for all n in ind
       If v is shorter than mask it will be repeated as necessary.
       In particular v can be a scalar or length 1 array.
       The routine put is the equivalent of the following (although the loop
       is in C for speed):

           ind = array(indices, copy=False)
           v = array(values, copy=False).astype(a, a.dtype)
           for i in ind: a.flat[i] = v[i]
       a must be a contiguous Numeric array.
    """
    return a.put(v,ind)

def putmask (a, mask, v):
    """putmask(a, mask, v) results in a = v for all places mask is true.
       If v is shorter than mask it will be repeated as necessary.
       In particular v can be a scalar or length 1 array.
    """
    return a.putmask(v, mask)

def swapaxes(a, axis1, axis2):
    """swapaxes(a, axis1, axis2) returns array a with axis1 and axis2
    interchanged.
    """
    a = array(a, copy=False)
    return a.swapaxes(axis1, axis2)

def transpose(a, axes=None):
    """transpose(a, axes=None) returns array with dimensions permuted
    according to axes.  If axes is None (default) returns array with
    dimensions reversed.
    """
    a = array(a,copy=False)
    return a.transpose(axes)

def sort(a, axis=-1):
    """sort(a,axis=-1) returns array with elements sorted along given axis.
    """
    a = array(a, copy=True)
    a.sort(axis)
    return a

def argsort(a, axis=-1):
    """argsort(a,axis=-1) return the indices into a of the sorted array
    along the given axis, so that take(a,result,axis) is the sorted array.
    """
    a = array(a, copy=False)
    return a.argsort(axis)

def argmax(a, axis=-1):
    """argmax(a,axis=-1) returns the indices to the maximum value of the
    1-D arrays along the given axis.
    """
    a = array(a, copy=False)
    return a.argmax(axis)

def argmin(a, axis=-1):
    """argmin(a,axis=-1) returns the indices to the minimum value of the
    1-D arrays along the given axis.
    """
    a = array(a,copy=False)
    return a.argmin(axis)

def searchsorted(a, v):
    """searchsorted(a, v)
    """
    a = array(a,copy=False)
    return a.searchsorted(v)

def resize(a, new_shape):
    """resize(a,new_shape) returns a new array with the specified shape.
    The original array's total size can be any size. It
    fills the new array with repeated copies of a.

    Note that a.resize(new_shape) will fill array with 0's
    beyond current definition of a.
    """

    a = ravel(a)
    Na = len(a)
    if not Na: return mu.zeros(new_shape, a.dtypechar)
    total_size = um.multiply.reduce(new_shape)
    n_copies = int(total_size / Na)
    extra = total_size % Na

    if extra != 0:
        n_copies = n_copies+1
        extra = Na-extra

    a = concatenate( (a,)*n_copies)
    if extra > 0:
        a = a[:-extra]

    return reshape(a, new_shape)

def squeeze(a):
    "Returns a with any ones from the shape of a removed"
    return asarray(a).squeeze()

def diagonal(a, offset=0, axis1=0, axis2=1):
    """diagonal(a, offset=0, axis1=0, axis2=1) returns the given diagonals
    defined by the last two dimensions of the array.
    """
    return asarray(a).diagonal(offset, axis1, axis2)

def trace(a, offset=0, axis1=0, axis2=1, dtype=None):
    """trace(a,offset=0, axis1=0, axis2=1) returns the sum along diagonals
    (defined by the last two dimenions) of the array.
    """
    return asarray(a).trace(offset, axis1, axis2, dtype)

def ravel(m):
    """ravel(m) returns a 1d array corresponding to all the elements of it's
    argument.
    """
    return asarray(m).ravel()

def nonzero(a):
    """nonzero(a) returns the indices of the elements of a which are not zero,
    a must be 1d
    """
    return asarray(a).nonzero()

def shape(a):
    """shape(a) returns the shape of a (as a function call which
       also works on nested sequences).
    """
    return asarray(a).shape

def compress(condition, m, axis=-1):
    """compress(condition, x, axis=-1) = those elements of x corresponding 
    to those elements of condition that are "true".  condition must be the
    same size as the given dimension of x."""
    return asarray(m).compress(condition, axis)

def clip(m, m_min, m_max):
    """clip(m, m_min, m_max) = every entry in m that is less than m_min is
    replaced by m_min, and every entry greater than m_max is replaced by
    m_max.
    """
    return asarray(m).clip(m_min, m_max)

def sum(x, axis=0, dtype=None):
    """Sum the array over the given axis.  The optional dtype argument
    is the data type for intermediate calculations.

    The default is to upcast (promote) smaller integer types to the
    platform-dependent Int.  For example, on 32-bit platforms:

        x.dtype                         default sum() dtype
        ---------------------------------------------------
        bool, Int8, Int16, Int32        Int32

    Examples:
    >>> sum([0.5, 1.5])
    2.0
    >>> sum([0.5, 1.5], dtype=Int32)
    1
    >>> sum([[0, 1], [0, 5]])
    array([0, 6])
    >>> sum([[0, 1], [0, 5]], axis=1)
    array([1, 5])
    """
    return asarray(x).sum(axis, dtype)

def product (x, axis=0, dtype=None):
    """Product of the array elements over the given axis."""
    return asarray(x).prod(axis, dtype)

def sometrue (x, axis=0):
    """Perform a logical_or over the given axis."""
    return asarray(x).any(axis)

def alltrue (x, axis=0):
    """Perform a logical_and over the given axis."""
    return asarray(x).all(axis)

def any(x,axis=None):
    """Return true if any elements of x are true:  sometrue(ravel(x))
    """
    return ravel(x).any(axis)

def all(x,axis=None):
    """Return true if all elements of x are true:  alltrue(ravel(x))
    """
    return ravel(x).all(axis)

def cumsum (x, axis=0, dtype=None):
    """Sum the array over the given axis."""
    return asarray(x).cumsum(axis, dtype)

def cumproduct (x, axis=0, dtype=None):
    """Sum the array over the given axis."""
    return asarray(x).cumprod(axis, dtype)

def ptp(a, axis=0):
    """Return maximum - minimum along the the given dimension
    """
    return asarray(a).ptp(axis)

def amax(a, axis=0):
    """Return the maximum of 'a' along dimension axis.
    """
    return asarray(a).max(axis)

def amin(a, axis=0):
    """Return the minimum of a along dimension axis.
    """
    return asarray(a).min(axis)

def alen(a):
    """Return the length of a Python object interpreted as an array
    """
    return len(asarray(a))

def prod(a, axis=0):
    """Return the product of the elements along the given axis
    """
    return asarray(a).prod(axis)

def cumprod(a, axis=0):
    """Return the cumulative product of the elments along the given axis
    """
    return asarray(a).cumprod(axis)

def ndim(a):
    try:
        return a.ndim
    except AttributeError:
        return asarray(a).ndim

def rank (a):
    """Get the rank of sequence a (the number of dimensions, not a matrix rank)
       The rank of a scalar is zero.
    """
    try:
        return a.ndim
    except AttributeError:
        return asarray(a).ndim

def size (a, axis=None):
    "Get the number of elements in sequence a, or along a certain axis."
    if axis is None:
        try:
            return a.size
        except AttributeError:
            return asarray(a).size
    else:
        try:
            return a.shape[axis]
        except AttributeError:
            return asarray(a).shape[axis]

def round_(a, decimals=0):
    """Round 'a' to the given number of decimal places.  Rounding
    behaviour is equivalent to Python.

    Return 'a' if the array is not floating point.  Round both the real
    and imaginary parts separately if the array is complex.
    """
    a = asarray(a)
    if not issubclass(a.dtype, _nx.inexact):
        return a
    if issubclass(a.dtype, _nx.complexfloating):
        return round_(a.real, decimals) + 1j*round_(a.imag, decimals)
    if decimals is not 0:
        decimals = asarray(decimals)
    s = sign(a)
    if decimals is not 0:
        a = absolute(multiply(a, 10.**decimals))
    else:
        a = absolute(a)
    rem = a-asarray(a).astype(_nx.intp)
    a = _nx.where(_nx.less(rem, 0.5), _nx.floor(a), _nx.ceil(a))
    # convert back
    if decimals is not 0:
        return multiply(a, s/(10.**decimals))
    else:
        return multiply(a, s)

around = round_

def mean(a, axis=0, dtype=None):
    return asarray(a).mean(axis, dtype)

def std(a, axis=0, dtype=None):
    return asarray(a).std(axis, dtype)

def var(a, axis=0, dtype=None):
    return asarray(a).var(axis, dtype)


"""\
Core FFT routines
==================

 Standard FFTs

   fft
   ifft
   fft2
   ifft2
   fftn
   ifftn

 Real FFTs

   refft
   irefft
   refft2
   irefft2
   refftn
   irefftn

 Hermite FFTs

   hfft
   ihfft
"""

depends = ['core']
global_symbols = ['fft','ifft']


__doc__ = \
""" Basic functions used by several sub-packages and useful to have in the
main name-space

Type handling
==============
iscomplexobj     --  Test for complex object, scalar result
isrealobj        --  Test for real object, scalar result
iscomplex        --  Test for complex elements, array result
isreal           --  Test for real elements, array result
imag             --  Imaginary part
real             --  Real part
real_if_close    --  Turns complex number with tiny imaginary part to real
isneginf         --  Tests for negative infinity ---|
isposinf         --  Tests for positive infinity    |
isnan            --  Tests for nans                 |----  array results
isinf            --  Tests for infinity             |
isfinite         --  Tests for finite numbers    ---| 
isscalar         --  True if argument is a scalar
nan_to_num       --  Replaces NaN's with 0 and infinities with large numbers
cast             --  Dictionary of functions to force cast to each type
common_type      --  Determine the 'minimum common type code' for a group
                       of arrays
mintypecode      --  Return minimal allowed common typecode.

Index tricks
==================
mgrid            --  Method which allows easy construction of N-d 'mesh-grids'
r_               --  Append and construct arrays: turns slice objects into
                       ranges and concatenates them, for 2d arrays appends
                       rows.
index_exp        --  Konrad Hinsen's index_expression class instance which
                     can be useful for building complicated slicing syntax.

Useful functions
==================
select           --  Extension of where to multiple conditions and choices
extract          --  Extract 1d array from flattened array according to mask
insert           --  Insert 1d array of values into Nd array according to mask
linspace         --  Evenly spaced samples in linear space
logspace         --  Evenly spaced samples in logarithmic space
fix              --  Round x to nearest integer towards zero
mod              --  Modulo mod(x,y) = x % y except keeps sign of y
amax             --  Array maximum along axis
amin             --  Array minimum along axis
ptp              --  Array max-min along axis
cumsum           --  Cumulative sum along axis
prod             --  Product of elements along axis
cumprod          --  Cumluative product along axis
diff             --  Discrete differences along axis
angle            --  Returns angle of complex argument
unwrap           --  Unwrap phase along given axis (1-d algorithm)
sort_complex     --  Sort a complex-array (based on real, then imaginary)
trim_zeros       --  trim the leading and trailing zeros from 1D array.

vectorize        --  a class that wraps a Python function taking scalar
                         arguments into a generalized function which
                         can handle arrays of arguments using the broadcast
                         rules of numerix Python.

Shape manipulation
===================
squeeze          --  Return a with length-one dimensions removed.
atleast_1d       --  Force arrays to be > 1D
atleast_2d       --  Force arrays to be > 2D
atleast_3d       --  Force arrays to be > 3D
vstack           --  Stack arrays vertically (row on row)
hstack           --  Stack arrays horizontally (column on column)
column_stack     --  Stack 1D arrays as columns into 2D array
dstack           --  Stack arrays depthwise (along third dimension)
split            --  Divide array into a list of sub-arrays
hsplit           --  Split into columns
vsplit           --  Split into rows
dsplit           --  Split along third dimension

Matrix (2d array) manipluations
===============================
fliplr           --  2D array with columns flipped
flipud           --  2D array with rows flipped
rot90            --  Rotate a 2D array a multiple of 90 degrees
eye              --  Return a 2D array with ones down a given diagonal
diag             --  Construct a 2D array from a vector, or return a given
                       diagonal from a 2D array.                       
mat              --  Construct a Matrix
bmat             --  Build a Matrix from blocks

Polynomials
============
poly1d           --  A one-dimensional polynomial class

poly             --  Return polynomial coefficients from roots
roots            --  Find roots of polynomial given coefficients
polyint          --  Integrate polynomial
polyder          --  Differentiate polynomial
polyadd          --  Add polynomials
polysub          --  Substract polynomials
polymul          --  Multiply polynomials
polydiv          --  Divide polynomials
polyval          --  Evaluate polynomial at given argument

Import tricks
=============
ppimport         --  Postpone module import until trying to use it
ppimport_attr    --  Postpone module import until trying to use its
                      attribute
ppresolve        --  Import postponed module and return it.

Machine arithmetics
===================
machar_single    --  MachAr instance storing the parameters of system
                     single precision floating point arithmetics
machar_double    --  MachAr instance storing the parameters of system
                     double precision floating point arithmetics

Threading tricks
================
ParallelExec     --  Execute commands in parallel thread.
"""

depends = ['core','testing']
global_symbols = ['*']


"""\
Core Linear Algebra Tools
===========

 Linear Algebra Basics:

   inv        --- Find the inverse of a square matrix
   solve      --- Solve a linear system of equations
   det        --- Find the determinant of a square matrix
   lstsq      --- Solve linear least-squares problem
   pinv       --- Pseudo-inverse (Moore-Penrose) using lstsq

 Eigenvalues and Decompositions:

   eig        --- Find the eigenvalues and vectors of a square matrix
   eigh       --- Find the eigenvalues and eigenvectors of a Hermitian matrix
   eigvals    --- Find the eigenvalues of a square matrix
   eigvalsh   --- Find the eigenvalues of a Hermitian matrix. 
   svd        --- Singular value decomposition of a matrix
   cholesky   --- Cholesky decomposition of a matrix   

"""

depends = ['core']



"""\
Core Random Tools
=================

"""

depends = ['core']
global_symbols = ['rand','randn']

__all__ = [
    'beta',
    'binomial',
    'bytes',
    'chisquare',
    'exponential',
    'f',
    'gamma',
    'geometric',
    'get_state',
    'gumbel',
    'hypergeometric',
    'laplace',
    'logistic',
    'lognormal',
    'logseries',
    'multinomial',
    'multivariate_normal',
    'negative_binomial',
    'noncentral_chisquare',
    'noncentral_f',
    'normal',
    'pareto',
    'permutation',
    'poisson',
    'power',
    'rand',
    'randint',
    'randn',
    'random_integers',
    'random_sample',
    'rayleigh',
    'seed',
    'set_state',
    'shuffle',
    'standard_cauchy',
    'standard_exponential',
    'standard_gamma',
    'standard_normal',
    'standard_t',
    'triangular',
    'uniform',
    'vonmises',
    'wald',
    'weibull',
    'zipf'
]



