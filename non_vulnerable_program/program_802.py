import f2pytest,sys
e = # expected
r = f2pytest.f()
if abs(r-e) > 1e-4:
    sys.stderr.write('expected %s but got %s\\n'%(e,r))
    sys.exit()
print 'ok'
"""
    tests.append(test)
#################################################################
if 0 or all:
    test={}
    test['name']='Trivial call to Fortran subroutine'
    test['id']='call'
    test['f']="""\
      subroutine f()
      end
"""
    test['py']="""\
