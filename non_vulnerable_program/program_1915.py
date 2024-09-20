        fromnumeric.round_(getdata(a), decimals, out)
        if hasattr(out, '_mask'):
            out._mask = getmask(a)
        return out

def arange(stop, start=None, step=1, dtype=None):
    "maskedarray version of the numpy function."
    return numpy.arange(stop, start, step, dtype).view(MaskedArray)
arange.__doc__ = numpy.arange.__doc__

def inner(a, b):
    "maskedarray version of the numpy function."
    fa = filled(a, 0)
    fb = filled(b, 0)
    if len(fa.shape) == 0:
        fa.shape = (1,)
    if len(fb.shape) == 0:
        fb.shape = (1,)
    return numpy.inner(fa, fb).view(MaskedArray)
inner.__doc__ = numpy.inner.__doc__
inner.__doc__ += "\n*Notes*:\n    Masked values are replaced by 0."
innerproduct = inner

def outer(a, b):
    "maskedarray version of the numpy function."
    fa = filled(a, 0).ravel()
    fb = filled(b, 0).ravel()
    d = numeric.outer(fa, fb)
    ma = getmask(a)
    mb = getmask(b)
    if ma is nomask and mb is nomask:
        return masked_array(d)
    ma = getmaskarray(a)
    mb = getmaskarray(b)
    m = make_mask(1-numeric.outer(1-ma, 1-mb), copy=0)
    return masked_array(d, mask=m)
outer.__doc__ = numpy.outer.__doc__
outer.__doc__ += "\n*Notes*:\n    Masked values are replaced by 0."
outerproduct = outer

def allequal (a, b, fill_value=True):
    """Returns True if all entries of  a and b are equal, using fill_value
as a truth value where either or both are masked.
    """
    m = mask_or(getmask(a), getmask(b))
    if m is nomask:
        x = getdata(a)
        y = getdata(b)
        d = umath.equal(x, y)
        return d.all()
    elif fill_value:
        x = getdata(a)
        y = getdata(b)
        d = umath.equal(x, y)
        dm = array(d, mask=m, copy=False)
        return dm.filled(True).all(None)
    else:
        return False

def allclose (a, b, fill_value=True, rtol=1.e-5, atol=1.e-8):
    """ Returns True if all elements of a and b are equal subject to given tolerances.
If fill_value is True, masked values are considered equal.
If fill_value is False, masked values considered unequal.
The relative error rtol should be positive and << 1.0
The absolute error atol comes into play for those elements of b that are very small
or zero; it says how small `a` must be also.
    """
    m = mask_or(getmask(a), getmask(b))
    d1 = getdata(a)
    d2 = getdata(b)
    x = filled(array(d1, copy=0, mask=m), fill_value).astype(float)
    y = filled(array(d2, copy=0, mask=m), 1).astype(float)
    d = umath.less_equal(umath.absolute(x-y), atol + rtol * umath.absolute(y))
    return fromnumeric.alltrue(fromnumeric.ravel(d))

#..............................................................................
def asarray(a, dtype=None):
    """asarray(data, dtype) = array(data, dtype, copy=0, subok=0)
Returns a as a MaskedArray object of the given dtype.
If dtype is not given or None, is is set to the dtype of a.
No copy is performed if a is already an array.
Subclasses are converted to the base class MaskedArray.
    """
    return masked_array(a, dtype=dtype, copy=False, keep_mask=True, subok=False)

def asanyarray(a, dtype=None):
    """asanyarray(data, dtype) = array(data, dtype, copy=0, subok=1)
Returns a as an masked array.
If dtype is not given or None, is is set to the dtype of a.
No copy is performed if a is already an array.
Subclasses are conserved.
    """
    return masked_array(a, dtype=dtype, copy=False, keep_mask=True, subok=True)


def empty(new_shape, dtype=float):
    "maskedarray version of the numpy function."
    return numpy.empty(new_shape, dtype).view(MaskedArray)
empty.__doc__ = numpy.empty.__doc__

def empty_like(a):
    "maskedarray version of the numpy function."
    return numpy.empty_like(a).view(MaskedArray)
empty_like.__doc__ = numpy.empty_like.__doc__

def ones(new_shape, dtype=float):
    "maskedarray version of the numpy function."
    return numpy.ones(new_shape, dtype).view(MaskedArray)
ones.__doc__ = numpy.ones.__doc__

def zeros(new_shape, dtype=float):
    "maskedarray version of the numpy function."
    return numpy.zeros(new_shape, dtype).view(MaskedArray)
zeros.__doc__ = numpy.zeros.__doc__

#####--------------------------------------------------------------------------
#---- --- Pickling ---
#####--------------------------------------------------------------------------
def dump(a,F):
    """Pickles the MaskedArray `a` to the file `F`.
`F` can either be the handle of an exiting file, or a string representing a file name.
    """
    if not hasattr(F,'readline'):
        F = open(F,'w')
    return cPickle.dump(a,F)

def dumps(a):
    """Returns a string corresponding to the pickling of the MaskedArray."""
    return cPickle.dumps(a)

def load(F):
    """Wrapper around ``cPickle.load`` which accepts either a file-like object or
 a filename."""
    if not hasattr(F, 'readline'):
        F = open(F,'r')
    return cPickle.load(F)

def loads(strg):
    "Loads a pickle from the current string."""
    return cPickle.loads(strg)


###############################################################################

if __name__ == '__main__':
