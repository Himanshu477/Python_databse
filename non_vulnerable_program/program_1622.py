from numpy.core.multiarray import ndarray
import numerictypes as _nt
import numpy as N
import sys as _sys

__all__ = ['NumArray']

class NumArray(ndarray):
    def __new__(klass, shape=None, type=None, buffer=None,
                byteoffset=0, bytestride=None, byteorder=_sys.byteorder,
                aligned=1, real=None, imag=None):

        type = _nt.getType(type)
        dtype = N.dtype(type._dtype)
        if byteorder in ['little', 'big']:
            if byteorder is not _sys.byteorder:
                dtype = dtype.newbyteorder()
        else:
            raise ValueError("byteorder must be 'little' or 'big'")

        if buffer is None:
            self = ndarray.__new__(klass, shape, dtype)
        else:
            self = ndarray.__new__(klass, shape, dtype, buffer=buffer,
                                   offset=byteoffset, strides=bytestride)
            
        self._type = type

        if real is not None:
            self.real = real

        if imag is not None:
            self.imag = imag

        self._byteorder = byteorder

        return self

    def argmax(self, axis=-1):
        return ndarray.argmax(self, axis)

    def argmin(self, axis=-1):
        return ndarray.argmax(self, axis)

    def argsort(self, axis=-1, kind='quicksort'):
        return ndarray.argmax(self, axis, kind)

    def astype(self, type=None):
        return self.astype(_getdtype(type))

    def byteswap(self):
        ndarray.byteswap(self, True)

    def byteswapped(self):
        return ndarray.byteswap(self, False)

    def getdtypechar(self):
        return self.dtype.char

    def getimag(self):
        return self.imag

    getimaginary = getimag

    imaginary = property(getimaginary, None, "")

    def getreal(self):
        return self.real

    def is_c_array(self):
        return self.dtype.isnative and self.flags.carray

    def is_f_array(self):
        return self.dtype.isnative and self.flags.farray
    
    def is_fortran_contiguous(self):
        return self.flags.contiguous

    def new(self, type=None):
        if type is not None:
            dtype = _getdtype(type)
            return N.empty(self.shape, dtype)
        else:
            return N.empty_like(self)

    def setimag(self, value):
        self.imag = value

    setimaginary = setimag

    def setreal(self, value):
        self.real = value

    def sinfo(self):
        self.info()

    def sort(self, axis=-1, kind='quicksort'):
        ndarray.sort(self, axis, kind)

    def spacesaver(self):
        return False

    def stddev(self):
        return self.std()

    def sum(self, type=None):
        dtype = _getdtype(type)
        return ndarray.sum(self, dtype=dtype)

    def togglebyteorder(self):
        self.dtype = self.dtype.newbyteorder()

    def type(self):
        return self._type

    def typecode(self):
        return _numtypecode[self.dtype.char]

    dtypechar = property(getdtypechar, None, "")
    
    def info(self):
        print "class: ", self.__class__
        print "shape: ", self.shape
        print "strides: ", self.strides
        print "byteoffset: 0"
        print "bytestride: ", self.strides[0]
        print "itemsize: ", self.itemsize
        print "aligned: ", self.flags.isaligned
        print "contiguous: ", self.flags.contiguous
        print "buffer: ", self.data
        print "data pointer:", self._as_paramater_
        print "byteorder: ", self._byteorder
        print "byteswap: ", not self.dtype.isnative


"""numerictypes: Define the numeric type objects

This module is designed so 'from numerictypes import *' is safe.
Exported symbols include:

  Dictionary with all registered number types (including aliases):
    typeDict

  Numeric type objects:
    Bool
    Int8 Int16 Int32 Int64
    UInt8 UInt16 UInt32 UInt64
    Float32 Double64
    Complex32 Complex64

  Numeric type classes:
    NumericType
      BooleanType
      SignedType
      UnsignedType
      IntegralType
        SignedIntegralType
        UnsignedIntegralType
      FloatingType
      ComplexType

$Id: numerictypes.py,v 1.55 2005/12/01 16:22:03 jaytmiller Exp $
"""

MAX_ALIGN = 8
MAX_INT_SIZE = 8

