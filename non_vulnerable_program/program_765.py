import f2pytest,sys
r = f2pytest.f('w')
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
    test['name']='Argument passing to Fortran function(character*9)'
    test['f']="""\
      function f(a)
      integer f
      character*9 a
      if (a .eq. 'abcdefgh ') then
          f = 3
      else
          write(*,*) "Fortran: expected 'abcdefgh ' but got '",a,"'"
          f = 4
      end if
      end
"""
    test['py']="""\
