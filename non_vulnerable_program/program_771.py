import f2pytest,sys
r = f2pytest.f(1+2j)
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
    for s in ['','*1','*2','*4','*8']:
        if s=='*8' and skip: continue
        test={}
        test['name']='Argument passing to Fortran function(logical%s)'%s
        test['f']="""\
      function f(a)
      integer f
      logical%s a
      if (a) then
          f = 3
      else
          write(*,*) "Fortran: expected .true. but got",a
          f = 4
      end if
      end
      function f2(a)
      integer f2
      logical%s a
      if (a) then
          write(*,*) "Fortran: expected .false. but got",a
          f2 = 4
      else
          f2 = 3
      end if
      end
"""%(s,s)
        test['py']="""\
