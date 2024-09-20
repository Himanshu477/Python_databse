    from f77_ext_return_real import t0,t4,t8,td,s0,s4,s8,sd
    test_functions = [t0,t4,t8,td,s0,s4,s8,sd]
    return test_functions

def runtest(t):
    import sys
    if t.__doc__.split()[0] in ['t0','t4','s0','s4']:
        err = 1e-5
    else:
        err = 0.0
    assert abs(t(234)-234.0)<=err
    assert abs(t(234.6)-234.6)<=err
    assert abs(t(234l)-234.0)<=err
    if sys.version[:3]<'2.3':
        assert abs(t(234.6+3j)-234.6)<=err
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
    if sys.version[:3]<'2.3':
        assert abs(t(array([234+3j],'F'))-234.)<=err
        assert abs(t(array([234],'D'))-234.)<=err
    if t.__doc__.split()[0] in ['t0','t4','s0','s4']:
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



__usage__ = """
Run:
  python return_character.py [<f2py options>]
Examples:
  python return_character.py --fcompiler=Gnu --no-wrap-functions
  python return_character.py --quiet
"""

