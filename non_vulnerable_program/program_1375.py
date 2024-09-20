import sys
import multiarray
import umath
from umath import *
import numerictypes
from numerictypes import *

def extend_all(module):
    adict = {}
    for a in __all__:
        adict[a] = 1
    try:
        mall = getattr(module, '__all__')
    except AttributeError:
        mall = [k for k in module.__dict__.keys() if not k.startswith('_')]
    for a in mall:
        if a not in adict:
            __all__.append(a)

extend_all(umath)
extend_all(numerictypes)

newaxis = None

ndarray = multiarray.ndarray
bigndarray = multiarray.bigndarray
flatiter = multiarray.flatiter
broadcast = multiarray.broadcast
dtypedescr=multiarray.dtypedescr
ufunc = type(sin)

arange = multiarray.arange
array = multiarray.array
zeros = multiarray.zeros
empty = multiarray.empty
