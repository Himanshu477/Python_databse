from arrayprint import array2string, get_printoptions, set_printoptions

_typelessdata = [int, float, complex]
if issubclass(intc, pyint):
    _typelessdata.append(intc)

def array_repr(arr, max_line_width=None, precision=None, suppress_small=None):
    if arr.size > 0 or arr.shape==(0,):
        lst = array2string(arr, max_line_width, precision, suppress_small,
                           ', ', "array(")
    else: # show zero-length shape unless it is (0,)
        lst = "[], shape=%s" % (repr(arr.shape),)
    typeless = arr.dtype in _typelessdata

    if arr.__class__ is not ndarray:
        cName= arr.__class__.__name__
    else:
        cName = "array"
    if typeless and arr.size:
        return cName + "(%s)" % lst
    else:
        typename=arr.dtype.__name__[:-8]
        return cName + "(%s, dtype=%s)" % (lst, typename)

def array_str(a, max_line_width = None, precision = None, suppress_small = None):
    return array2string(a, max_line_width, precision, suppress_small, ' ', "")

set_string_function = multiarray.set_string_function
set_string_function(array_str, 0)
set_string_function(array_repr, 1)

little_endian = (sys.byteorder == 'little')

def indices(dimensions, dtype=intp):
    """indices(dimensions,dtype=intp) returns an array representing a grid
    of indices with row-only, and column-only variation.
    """
    tmp = ones(dimensions, dtype)
    lst = []
    for i in range(len(dimensions)):
        lst.append( add.accumulate(tmp, i, )-1 )
    return array(lst)

def fromfunction(function, dimensions):
    """fromfunction(function, dimensions) returns an array constructed by
    calling function on a tuple of number grids.  The function should
    accept as many arguments as there are dimensions which is a list of
    numbers indicating the length of the desired output for each axis.
    """
    args = indices(dimensions)
    return function(*args)
    

