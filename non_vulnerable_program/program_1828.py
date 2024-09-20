import numpy as nx
from numpy import asarray

class error(Exception):
    pass

def array_set(vals1, indices, vals2):
    indices = asarray(indices)
    if indices.ndim != 1:
        raise ValueError, "index array must be 1-d"
    if not isinstance(vals1, ndarray):
        raise TypeError, "vals1 must be an ndarray"
    vals1 = asarray(vals1)
    vals2 = asarray(vals2)
    if vals1.ndim != vals2.ndim or vals1.ndim < 1:
        raise error, "vals1 and vals2 must have same number of dimensions (>=1)"
    vals1[indices] = vals2

def construct3(mask, itype):
    raise NotImplementedError

