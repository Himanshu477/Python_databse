    from f90_ext_return_integer import f90_return_integer as m
    test_functions = [m.t0,m.t1,m.t2,m.t4,m.t8,m.s0,m.s1,m.s2,m.s4,m.s8]
    return test_functions

def runtest(t):
    tname = t.__doc__.split()[0]
    assert t(123)==123
    assert t(123.6)==123
    assert t(123l)==123
    if sys.version[:3]<='2.2':
        assert t(123.6+3j)==123
    assert t('123')==123
    assert t(-123)==-123
    assert t([123])==123
    assert t((123,))==123
    assert t(array(123))==123
    assert t(array([123]))==123
    assert t(array([[123]]))==123
    assert t(array([123],'1'))==123
    assert t(array([123],'s'))==123
    assert t(array([123],'i'))==123
    assert t(array([123],'l'))==123
    assert t(array([123],'b'))==123
    assert t(array([123],'f'))==123
    assert t(array([123],'d'))==123
    if sys.version[:3]<='2.2':
        assert t(array([123+3j],'F'))==123
        assert t(array([123],'D'))==123

    try: raise RuntimeError,`t(array([123],'c'))`
    except ValueError: pass
    try: raise RuntimeError,`t('abc')`
    except ValueError: pass

    try: raise RuntimeError,`t([])`
    except IndexError: pass
    try: raise RuntimeError,`t(())`
    except IndexError: pass

    try: raise RuntimeError,`t(t)`
    except TypeError: pass
    try: raise RuntimeError,`t({})`
    except TypeError: pass

    if tname in ['t8','s8']:
        try: raise RuntimeError,`t(100000000000000000000000l)`
        except OverflowError: pass
        try: raise RuntimeError,`t(10000000011111111111111.23)`
        except OverflowError: pass
    else:
        if sys.version[:3]<='2.2':
            try: raise RuntimeError,`t(10000000000000l)`
            except OverflowError: pass
            try: raise RuntimeError,`t(10000000000.23)`
            except OverflowError: pass

if __name__=='__main__':
    #import libwadpy
    status = 1
    try:
        repeat,f2py_opts = f2py2e.f2py_testing.cmdline()
        test_functions = build(f2py_opts)
        f2py2e.f2py_testing.run(runtest,test_functions,repeat)
        print 'ok'
        status = 0
    finally:
        if status:
            print '*'*20
            print 'Running f2py2e.diagnose'
