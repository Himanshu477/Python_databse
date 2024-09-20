import f2pytest,sys
r = f2pytest.f(34)
if r==3:
    print 'ok'
elif r==4:
    sys.stderr.write('incorrect value received')
else:
    sys.stderr.write('incorrect return value')
"""
        tests.append(test)
################################################################
if 0 or all:
    test={}
    test['name']='Argument passing to Fortran function(integer*8)'
    test['f']="""\
      function f(a)
      integer f
      integer*8 a,e
      e = 20
      e = e*222222222
      if (a .eq. e) then
          f = 3
      else
          write(*,*) "Fortran: expected ",e," but got",a
          f = 4
      end if
      end
"""
    test['py']="""\
