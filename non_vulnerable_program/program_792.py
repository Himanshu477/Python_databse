import f2pytest,sys
e = 'Y'
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
    test['name']='Fortran function returning character*9'
    test['depends']=['fncall']
    test['f']="""\
      function f()
      character*9 f
      f = "abcdefgh"
      end
"""
    test['py']="""\
