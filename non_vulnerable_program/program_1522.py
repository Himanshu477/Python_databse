from cPickle import dump, dumps

multiarray = mu

def sarray(a, dtype=None, copy=False):
    return array(a, dtype, copy)

#Use this to add a new axis to an array
#compatibility only
NewAxis = None

#deprecated
UFuncType = type(um.sin)
UfuncType = type(um.sin)
ArrayType = mu.ndarray
arraytype = mu.ndarray

LittleEndian = (sys.byteorder == 'little')

# backward compatible names from old Precision.py

Character = 'S1'
UnsignedInt8 = _dt_(nt.uint8)
UInt8 = UnsignedInt8
UnsignedInt16 = _dt_(nt.uint16)
UInt16 = UnsignedInt16
UnsignedInt32 = _dt_(nt.uint32)
UInt32 = UnsignedInt32
UnsignedInt = _dt_(nt.uint)
UInt = UnsignedInt

try:
    UnsignedInt64 = _dt_(nt.uint64)
except AttributeError:
    pass
else:
    UInt64 = UnsignedInt64
    __all__ += ['UnsignedInt64', 'UInt64']
try:
    UnsignedInt128 = _dt_(nt.uint128)
except AttributeError:
    pass
else:
    UInt128 = UnsignedInt128
    __all__ += ['UnsignedInt128','UInt128']

Int8 = _dt_(nt.int8)
Int16 = _dt_(nt.int16)
Int32 = _dt_(nt.int32)

try:
    Int64 = _dt_(nt.int64)
except AttributeError:
    pass
else:
    __all__ += ['Int64']

try:
    Int128 = _dt_(nt.int128)
except AttributeError:
    pass
else:
    __all__ += ['Int128']

Bool = _dt_(bool)
Int0 = _dt_(int)
Int = _dt_(int)
Float0 = _dt_(float)
Float = _dt_(float)
Complex0 = _dt_(complex)
Complex = _dt_(complex)
PyObject = _dt_(nt.object_)
Float32 = _dt_(nt.float32)
Float64 = _dt_(nt.float64)

Float16='f'
Float8='f'
UnsignedInteger='L'
Complex8='F'
Complex16='F'

try:
    Float128 = _dt_(nt.float128)
except AttributeError:
    pass
else:
    __all__ += ['Float128']

Complex32 = _dt_(nt.complex64)
Complex64 = _dt_(nt.complex128)

try:
    Complex128 = _dt_(nt.complex256)
except AttributeError:
    pass
else:
    __all__ += ['Complex128']

def _deprecate(func, oldname, newname):
    import warnings
    def newfunc(*args,**kwds):
        warnings.warn("%s is deprecated, use %s" % (oldname, newname),
                      DeprecationWarning)
        return func(*args, **kwds)
    return newfunc

# backward compatibility
arrayrange = _deprecate(mu.arange, 'arrayrange', 'arange')
cross_correlate = _deprecate(correlate, 'cross_correlate', 'correlate')
cross_product = _deprecate(cross, 'cross_product', 'cross')
divide_safe = _deprecate(um.divide, 'divide_safe', 'divide')

# deprecated names
matrixmultiply = _deprecate(mu.dot, 'matrixmultiply', 'dot')
outerproduct = _deprecate(outer, 'outerproduct', 'outer')
innerproduct = _deprecate(mu.inner, 'innerproduct', 'inner')



def DumpArray(m, fp):
    m.dump(fp)

def LoadArray(fp):
    import cPickle
    return cPickle.load(fp)

def array_constructor(shape, typecode, thestr, Endian=LittleEndian):
    if typecode == "O":
        x = array(thestr, "O")
    else:
        x = mu.fromstring(thestr, typecode)
    x.shape = shape
    if LittleEndian != Endian:
        return x.byteswap(TRUE)
    else:
        return x




