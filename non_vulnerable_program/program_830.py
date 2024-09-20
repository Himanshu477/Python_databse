    from f90_ext_return_character import f90_return_char as m
    test_functions = [m.t0,m.t1,m.t5,m.ts,m.s0,m.s1,m.s5,m.ss]
    return test_functions


def runtest(t):
    tname = t.__doc__.split()[0]
    if tname in ['t0','t1','s0','s1']:
        assert t(23)=='2'
        r = t('ab');assert r=='a',`r`
        r = t(array('ab'));assert r=='a',`r`
        r = t(array(77,'1'));assert r=='M',`r`
        try: raise RuntimeError,`t(array([77,87]))`
        except ValueError: pass
        try: raise RuntimeError,`t(array(77))`
        except ValueError: pass
    elif tname in ['ts','ss']:
        assert t(23)=='23        ',`t(23)`
        assert t('123456789abcdef')=='123456789a',`t('123456789abcdef')`
    elif tname in ['t5','s5']:
        assert t(23)=='23   '
        assert t('ab')=='ab   '
        assert t('123456789abcdef')=='12345'
    else:
        raise NotImplementedError

if __name__=='__main__':
    #import libwadpy
    repeat,f2py_opts = f2py2e.f2py_testing.cmdline()
    test_functions = build(f2py_opts)
    f2py2e.f2py_testing.run(runtest,test_functions,repeat)
    print 'ok'



__usage__ = """
Run:
  python return_complex.py [<f2py options>]
Examples:
  python return_complex.py --quiet
"""


