import f2pytest,sys
r = f2pytest.f('abcdef5')
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
    for s in ['','*1','*2','*4']:
        test={}
        test['name']='Argument passing to Fortran function(integer%s)'%s
        test['f']="""\
      function f(a)
      integer f
      integer%s a
      if (a .eq. 34) then
          f = 3
      else
          write(*,*) "Fortran: expected 34 but got",a
          f = 4
      end if
      end
"""%s
        test['py']="""\
