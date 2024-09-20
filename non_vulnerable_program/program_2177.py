import numpy as N
import os
import ctypes

_path = os.path.dirname('__file__')
lib = N.ctypeslib.load_library('code', _path)
_typedict = {'zadd' : complex,
             'sadd' : N.single,
             'cadd' : N.csingle,
             'dadd' : float}
for name in _typedict.keys():
    val = getattr(lib, name)
    val.restype = None
    _type = _typedict[name]
    val.argtypes = [N.ctypeslib.ndpointer(_type, flags='aligned, contiguous'),
                    N.ctypeslib.ndpointer(_type, flags='aligned, contiguous'),
                    N.ctypeslib.ndpointer(_type, flags='aligned, contiguous,'\
                                          'writeable'),
                    N.ctypeslib.c_intp]

lib.dfilter2d.restype=None
lib.dfilter2d.argtypes = [N.ctypeslib.ndpointer(float, ndim=2,
                                                flags='aligned'),
                          N.ctypeslib.ndpointer(float, ndim=2,
                                                flags='aligned, contiguous,'\
                                                'writeable'),
                          ctypes.POINTER(N.ctypeslib.c_intp),
                          ctypes.POINTER(N.ctypeslib.c_intp)]        

def select(dtype):
    if dtype.char in ['?bBhHf']:
        return lib.sadd, N.single
    elif dtype.char in ['F']:
        return lib.cadd, N.csingle
    elif dtype.char in ['DG']:
        return lib.zadd, complex
    else:
        return lib.dadd, float
    return func, ntype

def add(a, b):
    requires = ['CONTIGUOUS', 'ALIGNED']
    a = N.asanyarray(a)
    func, dtype = select(a.dtype)
    a = N.require(a, dtype, requires)
    b = N.require(b, dtype, requires)
    c = N.empty_like(a)
    func(a,b,c,a.size)
    return c
 
def filter2d(a):
    a = N.require(a, float, ['ALIGNED'])
    b = N.zeros_like(a)
    lib.dfilter2d(a, b, a.ctypes.strides, a.ctypes.shape)
    return b
    




!    -*- f90 -*-
! Note: the context of this file is case sensitive.

python module add ! in 
    interface  ! in :add
        subroutine zadd(a,b,c,n) ! in :add:add.f
            double complex dimension(n) :: a
            double complex dimension(n) :: b
            double complex intent(out), dimension(n) :: c
            integer intent(hide), depend(a) :: n = len(a)
        end subroutine zadd
        subroutine cadd(a,b,c,n) ! in :add:add.f
            complex dimension(*) :: a
            complex dimension(*) :: b
            complex dimension(*) :: c
            integer :: n
        end subroutine cadd
        subroutine dadd(a,b,c,n) ! in :add:add.f
            double precision dimension(*) :: a
            double precision dimension(*) :: b
            double precision dimension(*) :: c
            integer :: n
        end subroutine dadd
        subroutine sadd(a,b,c,n) ! in :add:add.f
            real dimension(*) :: a
            real dimension(*) :: b
            real dimension(*) :: c
            integer :: n
        end subroutine sadd
    end interface 
end python module add

! This file was auto-generated with f2py (version:2_2694).
! See http://cens.ioc.ee/projects/f2py2e/


!    -*- f90 -*-
! Note: the context of this file is case sensitive.

python module filter ! in 
    interface  ! in :filter
        subroutine dfilter2d(a,b,m,n) ! in :filter:filter.f
            double precision dimension(m,n) :: a
            double precision dimension(m,n),intent(out),depend(m,n) :: b
            integer optional,intent(hide),check(shape(a,0)==m),depend(a) :: m=shape(a,0)
            integer optional,intent(hide),check(shape(a,1)==n),depend(a) :: n=shape(a,1)
        end subroutine dfilter2d
    end interface 
end python module filter

! This file was auto-generated with f2py (version:2_3032).
! See http://cens.ioc.ee/projects/f2py2e/


# -*- Mode: Python -*-  Not really, but close enough

cimport c_numpy
