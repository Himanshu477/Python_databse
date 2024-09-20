import f2pytest,sys
f2pytest.f()
print 'ok'
"""
    tests.append(test)
#################################################################
if 0 or all:
    test={}
    test['name']='Trivial call to Fortran function'
    test['f']="""\
      integer function f()
      f = 3
      end
"""
    test['py']="""\
