    from f90_ext_return_logical import f90_return_logical as m
    test_functions = [m.t0,m.t1,m.t2,m.t4,m.t8,m.s0,m.s1,m.s2,m.s4,m.s8]
    return test_functions




def runtest(t):
    assert t(True)==1,`t(True)`
    assert t(False)==0,`t(False)`
    assert t(0)==0
    assert t(None)==0
    assert t(0.0)==0
    assert t(0j)==0
    assert t(1j)==1
    assert t(234)==1
    assert t(234.6)==1
    assert t(234l)==1
    assert t(234.6+3j)==1
    assert t('234')==1
    assert t('aaa')==1
    assert t('')==0
    assert t([])==0
    assert t(())==0
    assert t({})==0
    assert t(t)==1
    assert t(-234)==1
    assert t(10l**100)==1
    assert t([234])==1
    assert t((234,))==1
    assert t(array(234))==1
    assert t(array([234]))==1
    assert t(array([[234]]))==1
    assert t(array([234],'1'))==1
    assert t(array([234],'s'))==1
    assert t(array([234],'i'))==1
    assert t(array([234],'l'))==1
    assert t(array([234],'b'))==1
    assert t(array([234],'f'))==1
    assert t(array([234],'d'))==1
    assert t(array([234+3j],'F'))==1
    assert t(array([234],'D'))==1
    assert t(array(0))==0
    assert t(array([0]))==0
    assert t(array([[0]]))==0
    assert t(array([0j]))==0
    assert t(array([1]))==1
    assert t(array([0,0]))==0
    assert t(array([0,1]))==1 #XXX: is this expected?

if __name__=='__main__':
    #import libwadpy
    repeat,f2py_opts = f2py2e.f2py_testing.cmdline()
    test_functions = build(f2py_opts)
    f2py2e.f2py_testing.run(runtest,test_functions,repeat)
    print 'ok'


