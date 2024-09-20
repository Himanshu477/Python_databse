    from numpy.distutils.core import setup
    setup(configuration=configuration)


"""
nary - convert integer to a number with an arbitrary base.
"""

__all__ = ['nary']

_alphabet='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
def _getalpha(r):
    if r>=len(_alphabet):
        return '_'+nary(r-len(_alphabet),len(_alphabet))
    return _alphabet[r]

def nary(number, base=64):
    """
    Return string representation of a number with a given base.
    """
    if isinstance(number, str):
        number = eval(number)
    n = number
    s = ''
    while n:
        n1 = n // base
        r = n - n1*base
        n = n1
        s = _getalpha(r) + s
    return s

def encode(string):
    import md5
    return nary('0x'+md5.new(string).hexdigest())

#print nary(12345124254252525522512324,64)


#!/usr/bin/env python
"""
Tests for intent(in,out) derived type arguments in Fortran subroutine's.

-----
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License. See http://scipy.org.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
Author: Pearu Peterson <pearu@cens.ioc.ee>
Created: Oct 2006
-----
"""

