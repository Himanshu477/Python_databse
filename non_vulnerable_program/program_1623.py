import numpy
LP64 = numpy.intp(0).itemsize == 8

HasUInt64 = 0
try:
    numpy.int64(0)
except:
    HasUInt64 = 0

#from typeconv import typeConverters as _typeConverters
#import numinclude
#from _numerictype import _numerictype, typeDict
