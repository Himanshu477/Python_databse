import f2pytest,sys
r = f2pytest.f()
if r:
    sys.stderr.write('expected .false. but got %s\\n'%(r))
r = f2pytest.f2()
if not r:
    sys.stderr.write('expected .true. but got %s\\n'%(r))
    sys.exit()
print 'ok'
"""
        tests.append(test)
################################################################
if 0 or all:
    test={}
    test['name']='Simple callback from Fortran'
    test['f']="""\
      subroutine f(g)
      external g
      call g()
      end
"""
    test['py']="""\
