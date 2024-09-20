from Numeric import *
import Numeric
import Matrix
from utility import isscalar
from fastumath import *


# Elementary Matrices

# zeros is from matrixmodule in C
# ones is from Numeric.py


def eye(N, M=None, k=0, typecode=None):
    """eye(N, M=N, k=0, typecode=None) returns a N-by-M matrix where the 
    k-th diagonal is all ones, and everything else is zeros.
    """
    if M is None: M = N
    if type(M) == type('d'): 
        typecode = M
        M = N
    m = equal(subtract.outer(arange(N), arange(M)),-k)
    if typecode is None:
        return m
    else:
        return m.astype(typecode)

def tri(N, M=None, k=0, typecode=None):
    """tri(N, M=N, k=0, typecode=None) returns a N-by-M matrix where all
    the diagonals starting from lower left corner up to the k-th are all ones.
    """
    if M is None: M = N
    if type(M) == type('d'): 
        typecode = M
        M = N
    m = greater_equal(subtract.outer(arange(N), arange(M)),-k)
    if typecode is None:
        return m
    else:
        return m.astype(typecode)
    
# Matrix manipulation

def diag(v, k=0):
    """diag(v,k=0) returns the k-th diagonal if v is a matrix or
    returns a matrix with v as the k-th diagonal if v is a vector.
    """
    v = asarray(v)
    s = v.shape
    if len(s)==1:
        n = s[0]+abs(k)
        if k > 0:
            v = concatenate((zeros(k, v.typecode()),v))
        elif k < 0:
            v = concatenate((v,zeros(-k, v.typecode())))
        return eye(n, k=k)*v
    elif len(s)==2:
        v = add.reduce(eye(s[0], s[1], k=k)*v)
        if k > 0: return v[k:]
        elif k < 0: return v[:k]
        else: return v
    else:
            raise ValueError, "Input must be 1- or 2-D."
    
def fliplr(m):
    """fliplr(m) returns a 2-D matrix m with the rows preserved and
    columns flipped in the left/right direction.  Only works with 2-D
    arrays.
    """
    m = asarray(m)
    if len(m.shape) != 2:
        raise ValueError, "Input must be 2-D."
    return m[:, ::-1]

def flipud(m):
    """flipud(m) returns a 2-D matrix with the columns preserved and
    rows flipped in the up/down direction.  Only works with 2-D arrays.
    """
    m = asarray(m)
    if len(m.shape) != 2:
        raise ValueError, "Input must be 2-D."
    return m[::-1]
    
# reshape(x, m, n) is not used, instead use reshape(x, (m, n))

def rot90(m, k=1):
    """rot90(m,k=1) returns the matrix found by rotating m by k*90 degrees
    in the counterclockwise direction.
    """
    m = asarray(m)
    if len(m.shape) != 2:
        raise ValueError, "Input must be 2-D."
    k = k % 4
    if k == 0: return m
    elif k == 1: return transpose(fliplr(m))
    elif k == 2: return fliplr(flipud(m))
    else: return fliplr(transpose(m))  # k==3

def tril(m, k=0):
    """tril(m,k=0) returns the elements on and below the k-th diagonal of
    m.  k=0 is the main diagonal, k > 0 is above and k < 0 is below the main
    diagonal.
    """
    svsp = m.spacesaver()
    m = asarray(m,savespace=1)
    out = tri(m.shape[0], m.shape[1], k=k, typecode=m.typecode())*m
    out.savespace(svsp)
    return out

def triu(m, k=0):
    """triu(m,k=0) returns the elements on and above the k-th diagonal of
    m.  k=0 is the main diagonal, k > 0 is above and k < 0 is below the main
    diagonal.
    """
    svsp = m.spacesaver()
    m = asarray(m,savespace=1)
    out = (1-tri(m.shape[0], m.shape[1], k-1, m.typecode()))*m
    out.savespace(svsp)
    return out

# Data analysis

# Basic operations
def amax(m,axis=-1):
    """Returns the maximum of m along dimension axis. 
    """
    if axis is None:
        m = ravel(m)
        axis = 0
    else:
        m = asarray(m)
    return maximum.reduce(m,axis)

def amin(m,axis=-1):
    """Returns the minimum of m along dimension axis.
    """
    if axis is None:
        m = ravel(m)
        axis = 0
    else:        
        m = asarray(m)
    return minimum.reduce(m,axis)

# Actually from Basis, but it fits in so naturally here...

def ptp(m,axis=-1):
    """Returns the maximum - minimum along the the given dimension
    """
    if axis is None:
        m = ravel(m)
        axis = 0
    else:
        m = asarray(m)
    return amax(m,axis)-amin(m,axis)

def cumsum(m,axis=-1):
    """Returns the cumulative sum of the elements along the given axis
    """
    if axis is None:
        m = ravel(m)
        axis = 0
    else:
        m = asarray(m)
    return add.accumulate(m,axis)

def prod(m,axis=-1):
    """Returns the product of the elements along the given axis
    """
    if axis is None:
        m = ravel(m)
        axis = 0
    else:
        m = asarray(m)
    return multiply.reduce(m,axis)

def cumprod(m,axis=-1):
    """Returns the cumulative product of the elments along the given axis
    """
    if axis is None:
        m = ravel(m)
        axis = 0
    else:
        m = asarray(m)
    return multiply.accumulate(m,axis)

def diff(x, n=1,axis=-1):
    """Calculates the nth order, discrete difference along given axis.
    """
    x = asarray(x)
    nd = len(x.shape)
    slice1 = [slice(None)]*nd
    slice2 = [slice(None)]*nd
    slice1[axis] = slice(1,None)
    slice2[axis] = slice(None,-1)
    if n > 1:
        return diff(x[slice1]-x[slice2], n-1, axis=axis)
    else:
        return x[slice1]-x[slice2]

def squeeze(a):
    "Returns a with any ones from the shape of a removed"
    a = asarray(a)
    b = asarray(a.shape)
    return reshape (a, tuple (compress (not_equal (b, 1), b)))

def sinc(x):
    """Returns sin(pi*x)/(pi*x) at all points of array x.
    """
    w = asarray(x*pi)
    return where(x==0, 1.0, sin(w)/w)

def angle(z,deg=0):
    """Return the angle of complex argument z."""
    if deg:
        fact = 180/pi
    else:
        fact = 1.0
    z = asarray(z)
    if z.typecode() in ['D','F']:
       zimag = z.imag
       zreal = z.real
    else:
       zimag = 0
       zreal = z
    return arctan2(zimag,zreal) * fact

