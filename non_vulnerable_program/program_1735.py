    from_param = classmethod(from_param)


# Factory for an array-checking class with from_param defined for
#  use with ctypes argtypes mechanism
_pointer_type_cache = {}
def ndpointer(dtype=None, ndim=None, shape=None, flags=None):
    if dtype is not None:
        dtype = _dtype(dtype)
    num = None
    if flags is not None:
        if isinstance(flags, str):
            flags = flags.split(',')
        elif isinstance(flags, (int, integer)):
            num = flags
            flags = _flags_fromnum(num)
        elif isinstance(flags, flagsobj):
            num = flags.num
            flags = _flags_fromnum(num)
        if num is None:
            try:
                flags = [x.strip().upper() for x in flags]
            except:
                raise TypeError, "invalid flags specification"
            num = _num_fromflags(flags)
    try:
        return _pointer_type_cache[(dtype, ndim, shape, num)]
    except KeyError:
        pass
    if dtype is None:
        name = 'any'
    elif dtype.names:
        name = str(id(dtype))
    else:
        name = dtype.str
    if ndim is not None:
        name += "_%dd" % ndim
    if shape is not None:
        try:
            strshape = [str(x) for x in shape]
        except TypeError:
            strshape = [str(shape)]
            shape = (shape,)
        shape = tuple(shape)
        name += "_"+"x".join(strshape)
    if flags is not None:
        name += "_"+"_".join(flags)
    else:
        flags = []
    klass = type("ndpointer_%s"%name, (_ndptr,),
                 {"_dtype_": dtype,
                  "_shape_" : shape,
                  "_ndim_" : ndim,
                  "_flags_" : num})
    _pointer_type_cache[dtype] = klass
    return klass


from numpy.testing import * 
set_package_path() 
import numpy as N 
from numpy.ctypeslib import ndpointer
restore_path() 
 
class test_ndpointer(NumpyTestCase): 
    def check_dtype(self): 
        dt = N.intc 
        p = ndpointer(dtype=dt) 
        self.assert_(p.from_param(N.array([1], dt))) 
        dt = '<i4' 
        p = ndpointer(dtype=dt) 
        self.assert_(p.from_param(N.array([1], dt))) 
        dt = N.dtype('>i4') 
        p = ndpointer(dtype=dt) 
        p.from_param(N.array([1], dt)) 
        self.assertRaises(TypeError, p.from_param, 
                          N.array([1], dt.newbyteorder('swap'))) 
        dtnames = ['x', 'y'] 
        dtformats = [N.intc, N.float64] 
        dtdescr = {'names' : dtnames, 'formats' : dtformats} 
        dt = N.dtype(dtdescr) 
        p = ndpointer(dtype=dt) 
        self.assert_(p.from_param(N.zeros((10,), dt))) 
        samedt = N.dtype(dtdescr) 
        p = ndpointer(dtype=samedt) 
        self.assert_(p.from_param(N.zeros((10,), dt))) 
        dt2 = N.dtype(dtdescr, align=True)
        if dt.itemsize != dt2.itemsize:
            self.assertRaises(TypeError, p.from_param, N.zeros((10,), dt2))
        else:
            self.assert_(p.from_param(N.zeros((10,), dt2)))
 
    def check_ndim(self): 
        p = ndpointer(ndim=0) 
        self.assert_(p.from_param(N.array(1))) 
        self.assertRaises(TypeError, p.from_param, N.array([1])) 
        p = ndpointer(ndim=1) 
        self.assertRaises(TypeError, p.from_param, N.array(1)) 
        self.assert_(p.from_param(N.array([1]))) 
        p = ndpointer(ndim=2) 
        self.assert_(p.from_param(N.array([[1]]))) 
         
    def check_shape(self): 
        p = ndpointer(shape=(1,2)) 
        self.assert_(p.from_param(N.array([[1,2]]))) 
        self.assertRaises(TypeError, p.from_param, N.array([[1],[2]])) 
        p = ndpointer(shape=()) 
        self.assert_(p.from_param(N.array(1))) 
 
    def check_flags(self): 
        x = N.array([[1,2,3]], order='F') 
        p = ndpointer(flags='FORTRAN') 
        self.assert_(p.from_param(x)) 
        p = ndpointer(flags='CONTIGUOUS') 
        self.assertRaises(TypeError, p.from_param, x) 
        p = ndpointer(flags=x.flags.num) 
        self.assert_(p.from_param(x)) 
        self.assertRaises(TypeError, p.from_param, N.array([[1,2,3]])) 
 
if __name__ == "__main__": 
    NumpyTest().run() 


"""
Discrete Fourier Transforms - FFT.py

The underlying code for these functions is an f2c translated and modified
version of the FFTPACK routines.

fft(a, n=None, axis=-1)
ifft(a, n=None, axis=-1)
rfft(a, n=None, axis=-1)
irfft(a, n=None, axis=-1)
hfft(a, n=None, axis=-1)
ihfft(a, n=None, axis=-1)
fftn(a, s=None, axes=None)
ifftn(a, s=None, axes=None)
rfftn(a, s=None, axes=None)
irfftn(a, s=None, axes=None)
fft2(a, s=None, axes=(-2,-1))
ifft2(a, s=None, axes=(-2, -1))
rfft2(a, s=None, axes=(-2,-1))
irfft2(a, s=None, axes=(-2, -1))
"""
__all__ = ['fft','ifft', 'rfft', 'irfft', 'hfft', 'ihfft', 'rfftn',
           'irfftn', 'rfft2', 'irfft2', 'fft2', 'ifft2', 'fftn', 'ifftn',
           'refft', 'irefft','refftn','irefftn', 'refft2', 'irefft2']

