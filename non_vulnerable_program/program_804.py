import f2pytest,sys
r = f2pytest.f()
if r==3:
    print 'ok'
else:
    sys.stderr.write('expected 3 but got %s'%r)
"""
    tests.append(test)

################################################################

if 0 or all:
    test={}
    test['name']='Simple callback from Fortran'
    test['id']='cb'
    test['f']="""\
      subroutine f(g)
      external g
      call g()
      end
"""
    test['py']="""\
