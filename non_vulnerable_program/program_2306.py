    import test2b_ext
    a = numpy.arange(1000)
    b = numpy.arange(1000)
    d = test2b_ext.add(a,b)

Except perhaps for the dependence on SWIG, Instant is a
straightforward utility for writing extension modules.


PyInline
--------

This is a much older module that allows automatic building of
extension modules so that C-code can be included with Python code.
It's latest release (version 0.03) was in 2001, and it appears that it
is not being updated.


PyFort
------

PyFort is a nice tool for wrapping Fortran and Fortran-like C-code
into Python with support for Numeric arrays. It was written by Paul
Dubois, a distinguished computer scientist and the very first
maintainer of Numeric (now retired). It is worth mentioning in the
hopes that somebody will update PyFort to work with NumPy arrays as
well which now support either Fortran or C-style contiguous arrays.


