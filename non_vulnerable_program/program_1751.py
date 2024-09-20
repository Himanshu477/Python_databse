    import foo
    print dir(foo)
    foo.foo(2,"abcdefg")


"""\
Core FFT routines
==================

 Standard FFTs

   fft
   ifft
   fft2
   ifft2
   fftn
   ifftn

 Real FFTs

   rfft
   irfft
   rfft2
   irfft2
   rfftn
   irfftn

 Hermite FFTs

   hfft
   ihfft
"""

depends = ['core']


"""
Generate
  int pyobj_to_<ctype>(PyObject* obj, <ctype>* value)
  PyObject* pyobj_from_<stype>(<ctype>* value)
functions.
"""
__all__ = ['pyobj_to_npy_scalar','pyobj_to_f2py_string','pyobj_from_npy_scalar']

