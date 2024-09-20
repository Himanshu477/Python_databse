import f2pytest,sys
r = f2pytest.f(34.56)
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
    for s in ['','*8','*16','*32']:
        if s=='*32' and skip: continue
        test={}
        test['name']='Argument passing to Fortran function(complex%s)'%s
        test['f']="""\
      function f(a)
      integer f
      complex%s a
      real*8 e
      e = abs(a-(1,2))
      if (e .lt. 1e-5) then
          f = 3
      else
          write(*,*) "Fortran: expected (1.,2.) but got",a
          f = 4
      end if
      end
"""%s
        test['py']="""\
