from cPickle import load, loads
_cload = load
_file = file

def load(file):
    if isinstance(file, type("")):
        file = _file(file,"rb")
    return _cload(file)

# These are all essentially abbreviations
# These might wind up in a special abbreviations module

def ones(shape, dtype=intp, fortran=0):
    """ones(shape, dtype=intp) returns an array of the given
    dimensions which is initialized to all ones. 
    """
    a=zeros(shape, dtype, fortran)
    a+=1
    ### a[...]=1  -- slower
    return a
 
def identity(n,dtype=intp):
    """identity(n) returns the identity matrix of shape n x n.
    """
    a = array([1]+n*[0],dtype=dtype)
    b = empty((n,n),dtype=dtype)
    b.flat = a
    return b

def allclose (a, b, rtol=1.e-5, atol=1.e-8):
    """ allclose(a,b,rtol=1.e-5,atol=1.e-8)
        Returns true if all components of a and b are equal
        subject to given tolerances.
        The relative error rtol must be positive and << 1.0
        The absolute error atol comes into play for those elements
        of y that are very small or zero; it says how small x must be also.
    """
    x = array(a, copy=0)
    y = array(b, copy=0)
    d = less(absolute(x-y), atol + rtol * absolute(y))
    return alltrue(ravel(d))
            

# Now a method....
##def setflags(arr, write=None, swap=None, uic=None, align=None):
##    if not isinstance(arr, ndarray):
##        raise ValueError, "first argument must be an array"
##    sdict = {}
##    if write is not None:
##        sdict['WRITEABLE'] = not not write
##    if swap is not None:
##        sdict['NOTSWAPPED'] = not swap
##    if uic is not None:
##        if (uic):
##            raise ValueError, "Can only set UPDATEIFCOPY flag to False"
##        sdict['UPDATEIFCOPY'] = False
##    if align is not None:
##        sdict['ALIGNED'] = not not align
##    arr.flags = sdict

_errdict = {"ignore":ERR_IGNORE,
            "warn":ERR_WARN,
            "raise":ERR_RAISE,
            "call":ERR_CALL}

_errdict_rev = {}
for key in _errdict.keys():
    _errdict_rev[_errdict[key]] = key

def seterr(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
    maskvalue = (_errdict[divide] << SHIFT_DIVIDEBYZERO) + \
                (_errdict[over] << SHIFT_OVERFLOW ) + \
                (_errdict[under] << SHIFT_UNDERFLOW) + \
                (_errdict[invalid] << SHIFT_INVALID)
    frame = sys._getframe().f_back
    frame.f_locals[UFUNC_ERRMASK_NAME] = maskvalue
    return

def geterr():
    frame = sys._getframe().f_back
    try:
        maskvalue = frame.f_locals[UFUNC_ERRMASK_NAME]
    except KeyError:
        maskvalue = ERR_DEFAULT

    mask = 3
    res = {}
    val = (maskvalue >> SHIFT_DIVIDEBYZERO) & mask
    res['divide'] = _errdict_rev[val]
    val = (maskvalue >> SHIFT_OVERFLOW) & mask
    res['over'] = _errdict_rev[val]
    val = (maskvalue >> SHIFT_UNDERFLOW) & mask
    res['under'] = _errdict_rev[val]
    val = (maskvalue >> SHIFT_INVALID) & mask
    res['invalid'] = _errdict_rev[val]
    return res

    
    


# Borrowed and adapted from numarray

"""numerictypes: Define the numeric type objects

This module is designed so 'from numeric3types import *' is safe.
Exported symbols include:

  Dictionary with all registered number types (including aliases):
    typeDict

  Type objects (not all will be available, depends on platform):
      see variable arraytypes for which ones you have

    Bit-width names
    
    int8 int16 int32 int64 int128
    uint8 uint16 uint32 uint64 uint128
    float16 float32 float64 float96 float128 float256
    complex32 complex64 complex128 complex192 complex256 complex512

    c-based names 

    bool

    object

    void, string, unicode 

    byte, ubyte,
    short, ushort
    intc, uintc,
    intp, uintp,
    int, uint,
    longlong, ulonglong,

    single, csingle,
    float, complex,
    longfloat, clongfloat,

    As part of the type-hierarchy:    xx -- is bit-width
    
     generic
       bool
       numeric
         integer
           signedinteger   (intxx)
             byte
             short
             int
             intp           int0
             longint
             longlong
           unsignedinteger  (uintxx)
             ubyte
             ushort
             uint
             uintp          uint0
             ulongint
             ulonglong
         floating           (floatxx)
             single          
             float  (double)
             longfloat
         complexfloating    (complexxx)
             csingle        
             complex (cfloat, cdouble)
             clongfloat
   
       flexible
         character
           string
           unicode
         void
   
       object

$Id: numerictypes.py,v 1.17 2005/09/09 22:20:06 teoliphant Exp $
"""

