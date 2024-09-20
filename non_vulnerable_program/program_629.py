fromstring = multiarray.fromstring
fromfile = multiarray.fromfile
frombuffer = multiarray.frombuffer
where = multiarray.where
concatenate = multiarray.concatenate
#def where(condition, x=None, y=None):
#    """where(condition,x,y) is shaped like condition and has elements of x and
#    y where condition is respectively true or false.
#    """
#    if (x is None) or (y is None):
#        return nonzero(condition)
#    return choose(not_equal(condition, 0), (y, x))

def asarray(a, dtype=None):
    """asarray(a,dtype=None) returns a as a NumPy array.  Unlike array(),
    no copy is performed if a is already an array.
    """
    return array(a, dtype, copy=0)

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
    return correlate(a,asarray(v)[::-1],mode)

ndarray = multiarray.ndarray
ufunc = type(sin)

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
