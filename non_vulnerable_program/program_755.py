import f2pytest,sys
def g():
    return 222222222L
r = f2pytest.f(g)
if not r==222222222L:
    sys.stderr.write('expected 222222222 but got %s\\n'%r)
    sys.exit()
print 'ok'
"""
    tests.append(test)
#################################################################
if 0 or all:
    for s in ['','*4','*8','*16']:
        if s=='*16' and skip: continue
        test={}
        test['name']='Callback function returning real%s'%s
        test['depends']=['fncall','cb']
        test['f']="""\
      function f(g)
      real%s f,g
      external g
      f=g()
      end
"""%s
        test['py']="""\
