import copy
def unwrap(p,discont=pi,axis=-1):
    """unwrap(p,discont=pi,axis=-1)

    unwraps radian phase p by changing absolute jumps greater than discont to
    their 2*pi complement along the given axis.
    """
    p = asarray(p)
    nd = len(p.shape)
    dd = diff(p,axis=axis)
    slice1 = [slice(None,None)]*nd     # full slices
    slice1[axis] = slice(1,None)
    ddmod = mod(dd+pi,2*pi)-pi
    putmask(ddmod,(ddmod==-pi) & (dd > 0),pi)
    ph_correct = ddmod - dd;
    putmask(ph_correct,abs(dd)<discont,0)
    up = array(p,copy=1,typecode='d')
    up[slice1] = p[slice1] + cumsum(ph_correct,axis)
    return up
 


def real_if_close(a,tol=1e-13):
    a = Numeric.asarray(a)
    if a.typecode() in ['F','D'] and Numeric.allclose(a.imag, 0, atol=tol):
        a = a.real
    return a

def sort_complex(a):
    """ Doesn't currently work for integer arrays -- only float or complex.
    """
    a = asarray(a,typecode=a.typecode().upper())
    def complex_cmp(x,y):
        res = cmp(x.real,y.real)
        if res == 0:
            res = cmp(x.imag,y.imag)
        return res
    l = a.tolist()                
    l.sort(complex_cmp)
    return array(l)




def test(level=10):
    from scipy_test import module_test
    module_test(__name__,__file__,level=level)

def test_suite(level=1):
    from scipy_test import module_test_suite
    return module_test_suite(__name__,__file__,level=level)





