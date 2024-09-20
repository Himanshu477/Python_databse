import f2pytest,sys
r = f2pytest.f(1)
if r==4:
    sys.stderr.write('incorrect value received')
    sys.exit()
elif not r==3:
    sys.stderr.write('incorrect return value')
    sys.exit()
r = f2pytest.f2(0)
if r==4:
    sys.stderr.write('incorrect value received')
    sys.exit()
elif not r==3:
    sys.stderr.write('incorrect return value')
    sys.exit()
print 'ok'
"""
        tests.append(test)


all = 1
tests = []
skip=1
#################################################################
if 0 or all:
    for s in ['','*1','*2','*4']:
        test={}
        test['name']='Callback function returning integer%s'%s
        test['depends']=['fncall','cb']
        test['f']="""\
      function f(g)
      integer%s f,g
      external g
      f=g()
      end
"""%s
        test['py']="""\
