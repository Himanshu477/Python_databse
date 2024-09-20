import f2pytest,sys
def g():
    return 'abcdefghi'
r = f2pytest.f(g)
if not r=='abcdefghi':
    sys.stderr.write('expected "abcdefghi" but got %s\\n'%r)
    sys.exit()
print 'ok'
"""
    tests.append(test)
#################################################################
if 0 or (all and not skip):
    test={}
    test['name']='Callback function returning character*(*)'
    test['f']="""\
      function f(g)
      character*(*) f,g
      external g
      f = g()
      end
"""
    test['py']="""\
