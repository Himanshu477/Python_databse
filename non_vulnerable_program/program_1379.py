from cPickle import load, loads
_cload = load
_file = file

def load(file):
    if isinstance(file, type("")):
        file = _file(file,"rb")
    return _cload(file)

# These are all essentially abbreviations
# These might wind up in a special abbreviations module

def ones(shape, dtype=int_, fortran=False):
    """ones(shape, dtype=int_) returns an array of the given
    dimensions which is initialized to all ones. 
    """
    # This appears to be slower...
    #a = empty(shape, dtype, fortran)
    #a.fill(1)
    a = zeros(shape, dtype, fortran)
    a+=1
    return a

def identity(n,dtype=int_):
    """identity(n) returns the identity matrix of shape n x n.
    """
    a = array([1]+n*[0],dtype=dtype)
    b = empty((n,n),dtype=dtype)
    b.flat = a
    return b

def allclose (a, b, rtol=1.e-5, atol=1.e-8):
    """ allclose(a,b,rtol=1.e-5,atol=1.e-8)
        Returns true if all components of a and b are equal
        subject to given tolerances.
        The relative error rtol must be positive and << 1.0
        The absolute error atol comes into play for those elements
        of y that are very small or zero; it says how small x must be also.
    """
    x = array(a, copy=False)
    y = array(b, copy=False)
    d = less(absolute(x-y), atol + rtol * absolute(y))
    return d.ravel().all()

def _setpyvals(lst, frame, where=0):
    if not isinstance(lst, list) or len(lst) != 3:
        raise ValueError, "Invalid pyvalues (length 3 list needed)."

    try:
        wh = where.lower()[0]
    except (AttributeError, TypeError, IndexError):
        wh = None

    if where==0 or wh == 'l':
        frame.f_locals[UFUNC_PYVALS_NAME] = lst
    elif where == 1 or wh == 'g':
        frame.f_globals[UFUNC_PYVALS_NAME] = lst
    elif where == 2 or wh == 'b':
        frame.f_builtins[UFUNC_PYVALS_NAME] = lst 

    umath.update_use_defaults()
    return

def _getpyvals(frame):
    try:
        return frame.f_locals[UFUNC_PYVALS_NAME]
    except KeyError:
        try:
            return frame.f_globals[UFUNC_PYVALS_NAME]
        except KeyError:
            try:
                return frame.f_builtins[UFUNC_PYVALS_NAME]
            except KeyError:
                return [UFUNC_BUFSIZE_DEFAULT, ERR_DEFAULT, None]

_errdict = {"ignore":ERR_IGNORE,
            "warn":ERR_WARN,
            "raise":ERR_RAISE,
            "call":ERR_CALL}

_errdict_rev = {}
for key in _errdict.keys():
    _errdict_rev[_errdict[key]] = key
del key

def seterr(divide="ignore", over="ignore", under="ignore",
           invalid="ignore", where=0):
    maskvalue = ((_errdict[divide] << SHIFT_DIVIDEBYZERO) +
                 (_errdict[over] << SHIFT_OVERFLOW ) +
                 (_errdict[under] << SHIFT_UNDERFLOW) +
                 (_errdict[invalid] << SHIFT_INVALID))

    frame = sys._getframe().f_back
    pyvals = _getpyvals(frame)
    pyvals[1] = maskvalue
    _setpyvals(pyvals, frame, where)

def geterr():
    frame = sys._getframe().f_back
    maskvalue = _getpyvals(frame)[1]

    mask = 3
    res = {}
    val = (maskvalue >> SHIFT_DIVIDEBYZERO) & mask
    res['divide'] = _errdict_rev[val]
    val = (maskvalue >> SHIFT_OVERFLOW) & mask
    res['over'] = _errdict_rev[val]
    val = (maskvalue >> SHIFT_UNDERFLOW) & mask
    res['under'] = _errdict_rev[val]
    val = (maskvalue >> SHIFT_INVALID) & mask
    res['invalid'] = _errdict_rev[val]
    return res

def setbufsize(size, where=0):
    if size > 10e6:
        raise ValueError, "Very big buffers.. %s" % size

    frame = sys._getframe().f_back
    pyvals = _getpyvals(frame)
    pyvals[0] = size
    _setpyvals(pyvals, frame, where)

def getbufsize():
    frame = sys._getframe().f_back
    return _getpyvals(frame)[0]

def seterrcall(func, where=0):
    if not callable(func):
        raise ValueError, "Only callable can be used as callback"
    frame = sys._getframe().f_back
    pyvals = _getpyvals(frame)
    pyvals[2] = func
    _setpyvals(pyvals, frame, where)

def geterrcall():
    frame = sys._getframe().f_back
    return _getpyvals(frame)[2]

def _setdef():
    frame = sys._getframe()
    defval = [UFUNC_BUFSIZE_DEFAULT, ERR_DEFAULT, None]
    frame.f_globals[UFUNC_PYVALS_NAME] = defval
    frame.f_builtins[UFUNC_PYVALS_NAME] = defval
    umath.update_use_defaults()

# set the default values
_setdef()

Inf = inf = infty = Infinity = PINF
nan = NaN = NAN

