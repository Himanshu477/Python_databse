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
if 0 and (all and not skip): 
    test={}
    test['name']='Callback function returning character*(*)' # not possible
    test['depends']=['fncall','cb']
    test['f']="""\
      function f(g)
      external g
      character*(*) g
      character*(*) f
      f = g()
      end
"""
    test['py']="""\
