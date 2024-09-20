import f2pytest,sys
e = 2
r = f2pytest.f()
if not r==e:
    sys.stderr.write('expected %s but got %s\\n'%(e,r))
    sys.exit()
print 'ok'
"""
        tests.append(test)
################################################################
if 0 or all:
    test={}
    test['name']='Fortran function returning integer*8'
    test['f']="""\
      function f()
      integer*8 f
      f = 20
      f = f * 222222222
      end
"""
    test['py']="""\
