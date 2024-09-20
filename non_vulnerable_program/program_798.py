import f2pytest,sys
e = 20L*222222222L
r = f2pytest.f()
if not r==e:
    sys.stderr.write('expected %s but got %s\\n'%(e,r))
    sys.exit()
print 'ok'
"""
    tests.append(test)
################################################################
if 0 or all:
    for s in ['','*1','*2','*4','*8']:
        test={}
        test['name']='Fortran function returning logical%s'%s
        test['depends']=['fncall']
        #test['f2pyflags']=['--debug-capi']
        test['f']="""\
      function f()
      logical%s f
      f = .false.
      end
      function f2()
      logical%s f2
      f2 = .true.
      end
"""%(s,s)
        test['py']="""\
