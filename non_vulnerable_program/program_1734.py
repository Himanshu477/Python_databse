from numpy.linalg import qr as _qr

def qr_decomposition(a, mode='full'):
    res = _qr(a, mode)
    if mode == 'full':
        return res
    return (None, res)










from numpy.testing import * 
set_package_path() 
import numpy as N 
restore_path() 
 
class test_ndpointer(NumpyTestCase): 
    def check_dtype(self): 
        dt = N.intc 
        p = N.ndpointer(dtype=dt) 
        self.assert_(p.from_param(N.array([1], dt))) 
        dt = '<i4' 
        p = N.ndpointer(dtype=dt) 
        self.assert_(p.from_param(N.array([1], dt))) 
        dt = N.dtype('>i4') 
        p = N.ndpointer(dtype=dt) 
        p.from_param(N.array([1], dt)) 
        self.assertRaises(TypeError, p.from_param, 
                          N.array([1], dt.newbyteorder('swap'))) 
        dtnames = ['x', 'y'] 
        dtformats = [N.intc, N.float64] 
        dtdescr = {'names' : dtnames, 'formats' : dtformats} 
        dt = N.dtype(dtdescr) 
        p = N.ndpointer(dtype=dt) 
        self.assert_(p.from_param(N.zeros((10,), dt))) 
        samedt = N.dtype(dtdescr) 
        p = N.ndpointer(dtype=samedt) 
        self.assert_(p.from_param(N.zeros((10,), dt))) 
        dt2 = N.dtype(dtdescr, align=True)
        if dt.itemsize != dt2.itemsize:
            self.assertRaises(TypeError, p.from_param, N.zeros((10,), dt2))
        else:
            self.assert_(p.from_param(N.zeros((10,), dt2)))
 
    def check_ndim(self): 
        p = N.ndpointer(ndim=0) 
        self.assert_(p.from_param(N.array(1))) 
        self.assertRaises(TypeError, p.from_param, N.array([1])) 
        p = N.ndpointer(ndim=1) 
        self.assertRaises(TypeError, p.from_param, N.array(1)) 
        self.assert_(p.from_param(N.array([1]))) 
        p = N.ndpointer(ndim=2) 
        self.assert_(p.from_param(N.array([[1]]))) 
         
    def check_shape(self): 
        p = N.ndpointer(shape=(1,2)) 
        self.assert_(p.from_param(N.array([[1,2]]))) 
        self.assertRaises(TypeError, p.from_param, N.array([[1],[2]])) 
        p = N.ndpointer(shape=()) 
        self.assert_(p.from_param(N.array(1))) 
 
    def check_flags(self): 
        x = N.array([[1,2,3]], order='F') 
        p = N.ndpointer(flags='FORTRAN') 
        self.assert_(p.from_param(x)) 
        p = N.ndpointer(flags='CONTIGUOUS') 
        self.assertRaises(TypeError, p.from_param, x) 
        p = N.ndpointer(flags=x.flags.num) 
        self.assert_(p.from_param(x)) 
        self.assertRaises(TypeError, p.from_param, N.array([[1,2,3]])) 
 
if __name__ == "__main__": 
    NumpyTest().run() 


__all__ = ['ctypes_load_library', 'ndpointer']


# Adapted from Albert Strasheim
def ctypes_load_library(libname, loader_path):
    if '.' not in libname:
        if sys.platform == 'win32':
            libname = '%s.dll' % libname
        elif sys.platform == 'darwin':
            libname = '%s.dylib' % libname
        else:
            libname = '%s.so' % libname
    loader_path = os.path.abspath(loader_path)
    if not os.path.isdir(loader_path):
        libdir = os.path.dirname(loader_path)
    else:
        libdir = loader_path
    import ctypes
    libpath = os.path.join(libdir, libname)
    return ctypes.cdll[libpath]

def _num_fromflags(flaglist):
    num = 0
    for val in flaglist:
        num += _flagdict[val]
    return num

def _flags_fromnum(num):
    res = []
    for key, value in _flagdict.items():
        if (num & value):
            res.append(key)
    return res

class _ndptr(object):
    def from_param(cls, obj):
        if not isinstance(obj, ndarray):
            raise TypeError, "argument must be an ndarray"
        if cls._dtype_ is not None \
               and obj.dtype != cls._dtype_:
            raise TypeError, "array must have data type %s" % cls._dtype_
        if cls._ndim_ is not None \
               and obj.ndim != cls._ndim_:
            raise TypeError, "array must have %d dimension(s)" % cls._ndim_
        if cls._shape_ is not None \
               and obj.shape != cls._shape_:
            raise TypeError, "array must have shape %s" % str(cls._shape_)
        if cls._flags_ is not None \
               and ((obj.flags.num & cls._flags_) != cls._flags_):
            raise TypeError, "array must have flags %s" % \
                  _flags_fromnum(cls._flags_)
        return obj.ctypes
