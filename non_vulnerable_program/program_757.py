import f2pytest,sys
def g():
    return 1+2j
r = f2pytest.f(g)
if abs(r-(1+2j))>1e-5:
    sys.stderr.write('expected (1+2j) but got %s\\n'%r)
    sys.exit()
print 'ok'
"""
        tests.append(test)
#################################################################
if 0 or all:
    for s in ['','*1','*2','*4','*8']:
        test={}
        test['name']='Callback function returning logical%s'%s
        test['depends']=['fncall','cb']
        test['f']="""\
      function f(g,h)
      logical%s f,g,h,a,b
      external g,h
      a=g()
      a=.not. a
      b=h()
      f = a .and. b
      end
"""%s
        test['py']="""\
