import f2pytest,sys
r = f2pytest.f(20L*222222222L)
if r==3:
    print 'ok'
elif r==4:
    sys.stderr.write('incorrect value received')
else:
    sys.stderr.write('incorrect return value')
"""
    tests.append(test)
#################################################################
if 0 or all:
    for s in ['','*4','*8','*16']:
        if s=='*16' and skip: continue
        test={}
        test['name']='Argument passing to Fortran function(real%s)'%s
        test['f']="""\
      function f(a)
      integer f
      real%s a,e
      e = abs(a-34.56)
      if (e .lt. 1e-5) then
          f = 3
      else
          write(*,*) "Fortran: expected 34.56 but got",a
          f = 4
      end if
      end
"""%s
        test['py']="""\
