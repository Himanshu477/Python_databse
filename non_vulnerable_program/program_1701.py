import numpy as N

oldtype2dtype = {'1': N.dtype(N.byte),
                 's': N.dtype(N.short),
                 'i': N.dtype(N.intc),
                 'l': N.dtype(int),
                 'b': N.dtype(N.ubyte),
                 'w': N.dtype(N.ushort),
                 'u': N.dtype(N.uintc),
                 'f': N.dtype(N.single),
                 'd': N.dtype(float),
                 'F': N.dtype(N.csingle),
                 'D': N.dtype(complex),
                 'O': N.dtype(object),
                 'c': N.dtype('c'),
                 None:N.dtype(int)
    }

def convtypecode(typecode, dtype=None):
    if dtype is None:
        try:
            dtype = oldtype2dtype[typecode]
        except:
            dtype = N.dtype(typecode)
    return dtype


__all__ = ['less', 'cosh', 'arcsinh', 'add', 'ceil', 'arctan2', 'floor_divide',
           'fmod', 'hypot', 'logical_and', 'power', 'sinh', 'remainder', 'cos',
           'equal', 'arccos', 'less_equal', 'divide', 'bitwise_or', 'bitwise_and',
           'logical_xor', 'log', 'subtract', 'invert', 'negative', 'log10', 'arcsin',
           'arctanh', 'logical_not', 'not_equal', 'tanh', 'true_divide', 'maximum',
           'arccosh', 'logical_or', 'minimum', 'conjugate', 'tan', 'greater', 'bitwise_xor',
           'fabs', 'floor', 'sqrt', 'arctan', 'right_shift', 'absolute', 'sin',
           'multiply', 'greater_equal', 'left_shift', 'exp']

