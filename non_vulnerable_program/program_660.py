from type_check import isreal

__all__.extend([key for key in dir(_nx.umath) \
                if key[0]!='_' and key not in __all__])

def _tocomplex(arr):
    if arr.dtypechar in ['f', 'h', 'B', 'b','H']:
        return arr.astype('F')
    else:
        return arr.astype('D')

def _fix_real_lt_zero(x):
    x = asarray(x)
    if any(isreal(x) & (x<0)):
        x = _tocomplex(x)
    return asscalar(x)

def _fix_real_abs_gt_1(x):
    x = asarray(x)
    if any(isreal(x) & (abs(x)>1)):
        x = _tocomplex(x)
    return x
    
def sqrt(x):
    x = _fix_real_lt_zero(x)
    return sqrt(x)

def log(x):
    x = _fix_real_lt_zero(x)
    return log(x)

def log10(x):
    x = _fix_real_lt_zero(x)
    return log10(x)    

def logn(n,x):
    """ Take log base n of x.
    """
    x = _fix_real_lt_zero(x)
    n = _fix_real_lt_zero(n)
    return log(x)/log(n)

def log2(x):
    """ Take log base 2 of x.
    """
    x = _fix_real_lt_zero(x)
    return log(x)/log(2)

def power(x, p):
    x = _fix_real_lt_zero(x)
    return power(x, p)


def arccos(x):
    x = _fix_real_abs_gt_1(x)
    return arccos(x)

def arcsin(x):
    x = _fix_real_abs_gt_1(x)
    return arcsin(x)

def arctanh(x):
    x = _fix_real_abs_gt_1(x)
    return arctanh(x)


"""
Unit-testing
------------

  ScipyTest -- Scipy tests site manager
  ScipyTestCase -- unittest.TestCase with measure method
  IgnoreException -- raise when checking disabled feature ('ignoring' is displayed)
  set_package_path -- prepend package build directory to path
  set_local_path -- prepend local directory (to tests files) to path
  restore_path -- restore path after set_package_path

Timing tools
------------

  jiffies -- return 1/100ths of a second that the current process has used
  memusage -- virtual memory size in bytes of the running python [linux]

Utility functions
-----------------

  assert_equal -- assert equality
  assert_almost_equal -- assert equality with decimal tolerance
  assert_approx_equal -- assert equality with significant digits tolerance
  assert_array_equal -- assert arrays equality
  assert_array_almost_equal -- assert arrays equality with decimal tolerance
  assert_array_less -- assert arrays less-ordering
  rand -- array of random numbers from given shape

"""

__all__ = []

