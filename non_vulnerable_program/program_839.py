    from f90_ext_return_real import f90_return_real as m
    test_functions = [m.t0,m.t4,m.t8,m.td,m.s0,m.s4,m.s8,m.sd]
    return test_functions

def runtest(t):
    tname =  t.__doc__.split()[0]
    if tname in ['t0','t4','s0','s4']:
        err = 1e-5
    else:
        err = 0.0
    assert abs(t(234)-234.0)<=err
    assert abs(t(234.6)-234.6)<=err
    assert abs(t(234l)-234.0)<=err
    if sys.version[:3]<='2.2':
        assert abs(t(234.6+3j)-234.6)<=err,`t(234.6+3j)`
    assert abs(t('234')-234)<=err
    assert abs(t('234.6')-234.6)<=err
    assert abs(t(-234)+234)<=err
    assert abs(t([234])-234)<=err
    assert abs(t((234,))-234.)<=err
    assert abs(t(array(234))-234.)<=err
    assert abs(t(array([234]))-234.)<=err
    assert abs(t(array([[234]]))-234.)<=err
    assert abs(t(array([234],'1'))+22)<=err
    assert abs(t(array([234],'s'))-234.)<=err
    assert abs(t(array([234],'i'))-234.)<=err
    assert abs(t(array([234],'l'))-234.)<=err
    assert abs(t(array([234],'b'))-234.)<=err
    assert abs(t(array([234],'f'))-234.)<=err
    assert abs(t(array([234],'d'))-234.)<=err
    if sys.version[:3]<='2.2':
        assert abs(t(array([234+3j],'F'))-234.)<=err,`t(array([234+3j],'F'))`
        assert abs(t(array([234],'D'))-234.)<=err,`t(array([234],'D'))`
    if tname in ['t0','t4','s0','s4']:
        assert t(1e200)==t(1e300) # inf

    try: raise RuntimeError,`t(array([234],'c'))`
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

    try:
        try: raise RuntimeError,`t(10l**400)`
        except OverflowError: pass
    except RuntimeError:
        r = t(10l**400); assert `r` in ['inf','Infinity'],`r`

if __name__=='__main__':
    #import libwadpy
    repeat,f2py_opts = f2py2e.f2py_testing.cmdline()
    test_functions = build(f2py_opts)
    f2py2e.f2py_testing.run(runtest,test_functions,repeat)
    print 'ok'


#!/usr/bin/env python
__usage__ = """
Run:
  python run.py [<f2py options>]
Examples:
  python run.py --quiet
"""

