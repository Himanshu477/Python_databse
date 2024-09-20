fromstring = multiarray.fromstring
fromfile = multiarray.fromfile
frombuffer = multiarray.frombuffer
newbuffer = multiarray.newbuffer
getbuffer = multiarray.getbuffer
where = multiarray.where
concatenate = multiarray.concatenate
fastCopyAndTranspose = multiarray._fastCopyAndTranspose
register_dtype = multiarray.register_dtype
set_numeric_ops = multiarray.set_numeric_ops
can_cast = multiarray.can_cast
lexsort = multiarray.lexsort


def asarray(a, dtype=None, fortran=False):
    """returns a as an array.  Unlike array(),
    no copy is performed if a is already an array.  Subclasses are converted
    to base class ndarray.
    """
    return array(a, dtype, copy=False, fortran=fortran)

def asanyarray(a, dtype=None, copy=False, fortran=False):
    """will pass subclasses through...
    """
    return array(a, dtype, copy=False, fortran=fortran, subok=1)

def isfortran(a):
    return a.flags['FNC']

# from Fernando Perez's IPython
def zeros_like(a):
    """Return an array of zeros of the shape and typecode of a.

    If you don't explicitly need the array to be zeroed, you should instead
    use empty_like(), which is faster as it only allocates memory."""
    a = asanyarray(a)
    return a.__array_wrap__(zeros(a.shape, a.dtype, a.flags['FNC']))

def empty_like(a):
    """Return an empty (uninitialized) array of the shape and typecode of a.

    Note that this does NOT initialize the returned array.  If you require
    your array to be initialized, you should use zeros_like().

    """
    a = asanyarray(a)
    return a.__array_wrap__(empty(a.shape, a.dtype, a.flags['FNC']))

# end Fernando's utilities

_mode_from_name_dict = {'v': 0,
                        's' : 1,
                        'f' : 2}

def _mode_from_name(mode):
    if isinstance(mode, type("")):
        return _mode_from_name_dict[mode.lower()[0]]
    return mode

def correlate(a,v,mode='valid'):
    mode = _mode_from_name(mode)
    return multiarray.correlate(a,v,mode)


def convolve(a,v,mode='full'):
    """Returns the discrete, linear convolution of 1-D
    sequences a and v; mode can be 0 (valid), 1 (same), or 2 (full)
    to specify size of the resulting sequence.
    """
    if (len(v) > len(a)):
        a, v = v, a
    mode = _mode_from_name(mode)
    return multiarray.correlate(a,asarray(v)[::-1],mode)


inner = multiarray.inner
dot = multiarray.dot

def outer(a,b):
    """outer(a,b) returns the outer product of two vectors.
    result(i,j) = a(i)*b(j) when a and b are vectors
    Will accept any arguments that can be made into vectors.
    """
    a = asarray(a)
    b = asarray(b)
    return a.ravel()[:,newaxis]*b.ravel()[newaxis,:]

def vdot(a, b):
    """Returns the dot product of 2 vectors (or anything that can be made into
    a vector). NB: this is not the same as `dot`, as it takes the conjugate
    of its first argument if complex and always returns a scalar."""
    return dot(asarray(a).ravel().conj(), asarray(b).ravel())

# try to import blas optimized dot if available
try:
    # importing this changes the dot function for basic 4 types
    # to blas-optimized versions.
