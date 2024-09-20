import f2pytest,sys
e = 'abcdefgh '
r = f2pytest.f()
if not e==r:
    sys.stderr.write('expected %s but got %s\\n'%(`e`,`r`))
    sys.exit()
print 'ok'
"""
    tests.append(test)
################################################################
if 0 or all:
    test={}
    test['name']='Fortran function returning character*(*)'
    test['f']="""\
      function f()
      character*(*) f
      f = "abcdefgh"
      end
"""
    test['py']="""\
