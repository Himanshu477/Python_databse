import f2pytest,sys
def g():
    return 4
r = f2pytest.f(g)
if not r==4:
    sys.stderr.write('expected 4 but got %s\\n'%r)
    sys.exit()
print 'ok'
"""
        tests.append(test)
#################################################################
if 0 or all:
    test={}
    test['name']='Callback function returning integer*8'
    test['f']="""\
      function f(g)
      integer*8 f,g
      external g
      f=g()
      end
"""
    test['py']="""\
