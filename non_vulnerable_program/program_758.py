import f2pytest,sys
def g():
    return 0
def h():
    return 1
r = f2pytest.f(g,h)
if not r:
    sys.stderr.write('expected .true. but got %s\\n'%r)
    sys.exit()
print 'ok'
"""
        tests.append(test)
#################################################################
if 0 or all:
    test={}
    #test['f2pyflags']=['--debug-capi']
    test['name']='Callback function returning character'
    test['depends']=['fncall','cb']
    test['f']="""\
      function f(g)
      character f
      character g
      external g
      f = g()
      end
"""
    test['py']="""\
