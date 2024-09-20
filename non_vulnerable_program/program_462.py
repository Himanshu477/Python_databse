import multiarray as mu
import umath as um
import numerictypes as nt
from numeric import asarray, array, correlate, outer
import sys

#Use this to add a new axis to an array

#compatibility only
NewAxis = None
#deprecated

UFuncType = type(um.sin)
ArrayType = mu.ndarray

LittleEndian = (sys.byteorder == 'little')


# backward compatible names from old Precision.py

Character = nt.string
UnsignedInt8 = nt.uint8
UnsignedInt16 = nt.uint16
UnsignedInt32 = nt.uint32
UnsignedInt = nt.uint


try:
    UnsignedInt64 = nt.uint64
    UnsignedInt128 = nt.uint128
except AttributeError:
    pass

Int8 = nt.int8
Int16 = nt.int16
Int32 = nt.int32

try:
    Int64 = nt.int64
    Int128 = nt.int128
except AttributeError:
    pass

Int0 = nt.intp
Float0 = nt.float
Float = nt.float
Complex0 = nt.complex
Complex = nt.complex
PyObject = nt.object

Float32 = nt.float32
Float64 = nt.float64

try:
    Float128 = nt.float128
except AttributeError:    
    pass

Complex32 = nt.complex64
Complex64 = nt.complex128

try:
    Complex128 = nt.complex256
except AttributeError:    
    pass

# backward compatibility
arrayrange = mu.arange
cross_correlate = correlate

# deprecated names
matrixmultiply = mu.dot
outerproduct=outer
innerproduct=mu.inner

