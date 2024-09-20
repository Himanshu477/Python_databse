from typeconv import oldtype2dtype as o2d

def eye(N, M=None, k=0, typecode=None):
    """ eye returns a N-by-M 2-d array where the  k-th diagonal is all ones,
        and everything else is zeros.
    """
    dtype = o2d[typecode]
    if M is None: M = N
    m = nn.equal(nn.subtract.outer(nn.arange(N), nn.arange(M)),-k)
    if m.dtype != dtype:
        return m.astype(dtype)
    
def tri(N, M=None, k=0, typecode=None):
    """ returns a N-by-M array where all the diagonals starting from
        lower left corner up to the k-th are all ones.
    """
    dtype = o2d[typecode]
    if M is None: M = N
    m = nn.greater_equal(nn.subtract.outer(nn.arange(N), nn.arange(M)),-k)
    if m.dtype != dtype:
        return m.astype(dtype)
    
def trapz(y, x=None, axis=-1):
    return _Ntrapz(y, x, axis=axis)

def ptp(x, axis=0):
    return _Nptp(x, axis)

def cumprod(x, axis=0):
    return _Ncumprod(x, axis)

def max(x, axis=0):
    return _Nmax(x, axis)

def min(x, axis=0):
    return _Nmin(x, axis)

def prod(x, axis=0):
    return _Nprod(x, axis)

def std(x, axis=0):
    return _Nstd(x, axis)

def mean(x, axis=0):
    return _Nmean(x, axis)

def cov(m, y=None, rowvar=0, bias=0):
    return _Ncov(m, y, rowvar, bias)

def corrcoef(x, y=None):
    return _Ncorrcoef(x,y,0,0)




# Backward compatible module for RandomArray

__all__ = ['ArgumentError','F','beta','binomial','chi_square', 'exponential', 'gamma', 'get_seed',
           'mean_var_test', 'multinomial', 'multivariate_normal', 'negative_binomial',
           'noncentral_F', 'noncentral_chi_square', 'normal', 'permutation', 'poisson', 'randint',
           'random', 'random_integers', 'seed', 'standard_normal', 'uniform']

ArgumentError = ValueError

