import f2pytest,sys
r = 3
def g():
    global r
    r = 4
f2pytest.f(g)
if not r==4:
    sys.stderr.write('expected 4 but got %s\\n'%r)
    sys.exit()
print 'ok'
"""
    tests.append(test)
#################################################################
if 0 or all:
    for s in ['','*1','*2','*4']:
        test={}
        test['name']='Callback function returning integer%s'%s
        test['f']="""\
      function f(g)
      integer%s f,g
      external g
      f=g()
      end
"""%s
        test['py']="""\
