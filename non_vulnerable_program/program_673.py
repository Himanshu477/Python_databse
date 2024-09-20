import calc_lwork

class LinAlgError(Exception):
    pass

# Linear equations
def solve(a, b, sym_pos=0, lower=0, overwrite_a=0, overwrite_b=0,
          debug = 0):
    """ solve(a, b, sym_pos=0, lower=0, overwrite_a=0, overwrite_b=0) -> x

    Solve a linear system of equations a * x = b for x.

    Inputs:

      a -- An N x N matrix.
      b -- An N x nrhs matrix or N vector.
      sym_pos -- Assume a is symmetric and positive definite.
      lower -- Assume a is lower triangular, otherwise upper one.
               Only used if sym_pos is true.
      overwrite_y - Discard data in y, where y is a or b.

    Outputs:

      x -- The solution to the system a * x = b
    """
    a1, b1 = map(asarray_chkfinite,(a,b))
    if len(a1.shape) != 2 or a1.shape[0] != a1.shape[1]:
        raise ValueError, 'expected square matrix'
    if a1.shape[0] != b1.shape[0]:
        raise ValueError, 'incompatible dimensions'
    overwrite_a = overwrite_a or (a1 is not a and not hasattr(a,'__array__'))
    overwrite_b = overwrite_b or (b1 is not b and not hasattr(b,'__array__'))
    if debug:
        print 'solve:overwrite_a=',overwrite_a
        print 'solve:overwrite_b=',overwrite_b
    if sym_pos:
        posv, = get_lapack_funcs(('posv',),(a1,b1),debug=debug)
        c,x,info = posv(a1,b1,
                        lower = lower,
                        overwrite_a=overwrite_a,
                        overwrite_b=overwrite_b)
    else:
        gesv, = get_lapack_funcs(('gesv',),(a1,b1))
        lu,piv,x,info = gesv(a1,b1,
                             overwrite_a=overwrite_a,
                             overwrite_b=overwrite_b)
        
    if info==0:
        return x
    if info>0:
        raise LinAlgError, "singular matrix"
    raise ValueError,\
          'illegal value in %-th argument of internal gesv|posv'%(-info)

# matrix inversion
def inv(a, overwrite_a=0):
    """ inv(a, overwrite_a=0) -> a_inv

    Return inverse of square matrix a.
    """
    pass

## matrix and Vector norm
