from numpy import dot as matrixmultiply, dot, vdot, ravel

def array(sequence=None, typecode=None, copy=1, savespace=0,
          type=None, shape=None, dtype=None):
    dtype = type2dtype(typecode, type, dtype)
    if sequence is None:
        if shape is None:
            return None
        if dtype is None:
            dtype = 'l'
        return N.empty(shape, dtype)
    arr = N.array(sequence, dtype, copy=copy)
    if shape is not None:
        arr.shape = shape
    return arr

def asarray(seq, type=None, typecode=None, dtype=None):
    if seq is None:
        return None
    dtype = type2dtype(typecode, type, dtype)
    return N.array(seq, dtype, copy=0)

def ones(shape, type=None, typecode=None, dtype=None):
    dtype = type2dtype(typecode, type, dtype)
    return N.ones(shape, dtype)

def zeros(shape, type=None, typecode=None, dtype=None):
    dtype = type2dtype(typecode, type, dtype)
    return N.zeros(shape, dtype)

def where(condition, x=None, y=None, out=None):
    if x is None and y is None:
        arr = N.where(condition)
    else:
        arr = N.where(condition, x, y)
    if out is not None:
        out[...] = arr
        return out
    return arr
    
def indices(shape, type=None):
    return N.indices(shape, type)

def arange(a1, a2=None, stride=1, type=None, shape=None,
           typecode=None, dtype=None):
    dtype = type2dtype(typecode, type, dtype)
    return N.arange(a1, a2, stride, dtype)



__all__ = ['abs', 'absolute', 'add', 'arccos', 'arccosh', 'arcsin', 'arcsinh',
           'arctan', 'arctan2', 'arctanh', 'bitwise_and', 'bitwise_not',
           'bitwise_or', 'bitwise_xor', 'ceil', 'cos', 'cosh', 'cumproduct',
           'cumsum', 'divide', 'equal', 'exp', 'fabs', 'floor', 'floor_divide',
           'fmod', 'greater', 'greater_equal', 'hypot', 'ieeemask', 'isnan',
           'less', 'less_equal', 'log', 'log10', 'logical_and', 'logical_not',
           'logical_or', 'logical_xor', 'lshift', 'maximum', 'minimum',
           'minus', 'multiply', 'negative', 'nonzero', 'not_equal',
           'power', 'product', 'remainder', 'rshift', 'sin', 'sinh', 'sqrt',
           'subtract', 'sum', 'tan', 'tanh', 'true_divide']

