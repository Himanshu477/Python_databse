import Numeric

def logspace(start,stop,num=50,endpoint=1):
    """Evenly spaced samples on a logarithmic scale.

    Return num evenly spaced samples from 10**start to 10**stop.  If
    endpoint=1 then last sample is 10**stop.
    """
    if endpoint:
        step = (stop-start)/float((num-1))
        y = Numeric.arange(0,num) * step + start
    else:
        step = (stop-start)/float(num)
        y = Numeric.arange(0,num) * step + start
    return Numeric.power(10.0,y)

def linspace(start,stop,num=50,endpoint=1,retstep=0):
    """Evenly spaced samples.
    
    Return num evenly spaced samples from start to stop.  If endpoint=1 then
    last sample is stop. If retstep is 1 then return the step value used.
    """
    if endpoint:
        step = (stop-start)/float((num-1))
        y = Numeric.arange(0,num) * step + start        
    else:
        step = (stop-start)/float(num)
        y = Numeric.arange(0,num) * step + start
    if retstep:
        return y, step
    else:
        return y

#def round(arr):
#    return Numeric.floor(arr+0.5)
round = Numeric.around
any = Numeric.sometrue
all = Numeric.alltrue

def fix(x):
    """Round x to nearest integer towards zero.
    """
    x = Numeric.asarray(x)
    y = Numeric.floor(x)
    return Numeric.where(x<0,y+1,y)

def mod(x,y):
    """x - y*floor(x/y)
    
    For numeric arrays, x % y has the same sign as x while
    mod(x,y) has the same sign as y.
    """
    return x - y*Numeric.floor(x*1.0/y)

def fftshift(x,axes=None):
    """Shift the result of an FFT operation.

    Return a shifted version of x (useful for obtaining centered spectra).
    This function swaps "half-spaces" for all axes listed (defaults to all)
    """
    ndim = len(x.shape)
    if axes == None:
        axes = range(ndim)
    y = x
    for k in axes:
        N = x.shape[k]
        p2 = int(Numeric.ceil(N/2.0))
        mylist = Numeric.concatenate((Numeric.arange(p2,N),Numeric.arange(p2)))
        y = Numeric.take(y,mylist,k)
    return y

def ifftshift(x,axes=None):
    """Reverse the effect of fftshift.
    """
    ndim = len(x.shape)
    if axes == None:
        axes = range(ndim)
    y = x
    for k in axes:
        N = x.shape[k]
        p2 = int(Numeric.floor(N/2.0))
        mylist = Numeric.concatenate((Numeric.arange(p2,N),Numeric.arange(p2)))
        y = Numeric.take(y,mylist,k)
    return y

def fftfreq(N,sample=1.0):
    """FFT sample frequencies
    
    Return the frequency bins in cycles/unit (with zero at the start) given a
    window length N and a sample spacing.
    """
    N = int(N)
    sample = float(sample)
    return Numeric.concatenate((Numeric.arange(0,(N-1)/2+1,1,'d'),Numeric.arange(-(N-1)/2,0,1,'d')))/N/sample

def cont_ft(gn,fr,delta=1.0,n=None):
    """Compute the (scaled) DFT of gn at frequencies fr.

    If the gn are alias-free samples of a continuous time function then the
    correct value for the spacing, delta, will give the properly scaled,
    continuous Fourier spectrum.

    The DFT is obtained when delta=1.0
    """
    if n is None:
        n = Numeric.arange(len(gn))
    dT = delta
    trans_kernel = Numeric.exp(-2j*Numeric.pi*fr[:,Numeric.NewAxis]*dT*n)
    return dT*Numeric.dot(trans_kernel,gn)

def toeplitz(c,r=None):
    """Construct a toeplitz matrix (i.e. a matrix with constant diagonals).

    Description:

       toeplitz(c,r) is a non-symmetric Toeplitz matrix with c as its first
       column and r as its first row.

       toeplitz(c) is a symmetric (Hermitian) Toeplitz matrix (r=c). 

    See also: hankel
    """
    if isscalar(c) or isscalar(r):
        return c   
    if r is None:
        r = c
        r[0] = Numeric.conjugate(r[0])
        c = Numeric.conjugate(c)
    r,c = map(Numeric.asarray,(r,c))
    r,c = map(Numeric.ravel,(r,c))
    rN,cN = map(len,(r,c))
    if r[0] != c[0]:
        print "Warning: column and row values don't agree; column value used."
    vals = r_[r[rN-1:0:-1], c]
    cols = grid[0:cN]
    rows = grid[rN:0:-1]
    indx = cols[:,Numeric.NewAxis]*Numeric.ones((1,rN)) + \
           rows[Numeric.NewAxis,:]*Numeric.ones((cN,1)) - 1
    return Numeric.take(vals, indx)


def hankel(c,r=None):
    """Construct a hankel matrix (i.e. matrix with constant anti-diagonals).

    Description:

      hankel(c,r) is a Hankel matrix whose first column is c and whose
      last row is r.

      hankel(c) is a square Hankel matrix whose first column is C.
      Elements below the first anti-diagonal are zero.

    See also:  toeplitz
    """
    if isscalar(c) or isscalar(r):
        return c   
    if r is None:
        r = Numeric.zeros(len(c))
    elif r[0] != c[-1]:
        print "Warning: column and row values don't agree; column value used."
    r,c = map(Numeric.asarray,(r,c))
    r,c = map(Numeric.ravel,(r,c))
    rN,cN = map(len,(r,c))
    vals = r_[c, r[1:rN]]
    cols = grid[1:cN+1]
    rows = grid[0:rN]
    indx = cols[:,Numeric.NewAxis]*Numeric.ones((1,rN)) + \
           rows[Numeric.NewAxis,:]*Numeric.ones((cN,1)) - 1
    return Numeric.take(vals, indx)


def real(val):
    aval = asarray(val)
    if aval.typecode() in ['F', 'D']:
        return aval.real
    else:
        return aval

def imag(val):
    aval = asarray(val)
    if aval.typecode() in ['F', 'D']:
        return aval.imag
    else:
        return array(0,aval.typecode())*aval

def iscomplex(x):
    return imag(x) != Numeric.zeros(asarray(x).shape)

def isreal(x):
    return imag(x) == Numeric.zeros(asarray(x).shape)

def array_iscomplex(x):
    return asarray(x).typecode() in ['F', 'D']

def array_isreal(x):
    return not asarray(x).typecode() in ['F', 'D']

def isposinf(val):
    # complex not handled currently (and potentially ambiguous)
    return Numeric.logical_and(isinf(val),val > 0)

def isneginf(val):
    # complex not handled currently (and potentially ambiguous)
    return Numeric.logical_and(isinf(val),val < 0)
    
def nan_to_num(x):
    # mapping:
    #    NaN -> 0
    #    Inf -> scipy.limits.double_max
    #   -Inf -> scipy.limits.double_min
    # complex not handled currently
    import scipy.limits
    try:
        t = x.typecode()
    except AttributeError:
        t = type(x)
    if t in [ComplexType,'F','D']:    
        y = nan_to_num(x.real) + 1j * nan_to_num(x.imag)
    else:    
        x = Numeric.asarray(x)
        are_inf = isposinf(x)
        are_neg_inf = isneginf(x)
        are_nan = isnan(x)
        choose_array = are_neg_inf + are_nan * 2 + are_inf * 3
        y = Numeric.choose(choose_array,
                   (x,scipy.limits.double_min, 0., scipy.limits.double_max))
    return y

# These are from Numeric
