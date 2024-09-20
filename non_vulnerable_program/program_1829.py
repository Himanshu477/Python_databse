from numpy import digitize

def find_mask(fs, node_edges):
    raise NotImplementedError

def histogram(lst, weight=None):
    raise NotImplementedError

def index_sort(arr):
    return asarray(arr).argsort(kind='heap')

def interp(y, x, z, typ=None):
    """y(z) interpolated by treating y(x) as piecewise function
    """
    res = numpy.interp(z, x, y)
    if typ is None or typ == 'd':
        return res
    if typ == 'f':
        return res.astype('f')

    raise error, "incompatible typecode"

def nz(x):
    x = asarray(x,dtype=nx.ubyte)
    if x.ndim != 1:
        raise TypeError, "intput must have 1 dimension."
    indxs = nx.flatnonzero(x != 0)
    return indxs[-1].item()+1

def reverse(x, n):
    x = asarray(x,dtype='d')
    if x.ndim != 2:
        raise ValueError, "input must be 2-d"
    y = nx.empty_like(x)
    if n == 0:
        y[...] = x[::-1,:]
    elif n == 1:
        y[...] = x[:,::-1]
    return y

def span(lo, hi, num, d2=0):
    x = linspace(lo, hi, num)
    if d2 <= 0
        return x
    else:
        ret = empty((d2,num),x.dtype)
        ret[...] = x
        return ret

def to_corners(arr, nv, nvsum):
    raise NotImplementedError

def zmin_zmax(z, ireg):
    raise NotImplementedError
        


major = 2

try:
