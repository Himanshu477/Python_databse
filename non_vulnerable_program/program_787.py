import f2pytest,sys
def g():
    return 't'
r = f2pytest.f(g)
if not r=='t':
    sys.stderr.write('expected "t" but got %s\\n'%r)
    sys.exit()
print 'ok'
"""
    tests.append(test)
#################################################################
if 0 or all:
    test={}
    test['name']='Callback function returning character*9'
    test['f']="""\
      function f(g)
      character*9 f,g
      external g
      f = g()
      end
"""
    test['py']="""\
