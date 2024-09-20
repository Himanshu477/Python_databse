import f2pytest,sys
r = f2pytest.f()
if r==3:
    print 'ok'
else:
    sys.stderr.write('expected 3 but got %s'%r)
"""
    tests.append(test)
#################################################################
if 0 or all:
    test={}
    test['name']='Argument passing to Fortran function(character)'
    test['f']="""\
      function f(a)
      integer f
      character a
      if (a .eq. 'w') then
          f = 3
      else
          write(*,*) "Fortran: expected 'w' but got '",a,"'"
          f = 4
      end if
      end
"""
    test['py']="""\
