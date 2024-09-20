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
#################################################################
if 0 or all:
    for s in ['','*1']:
        test={}
        test['name']='Fortran function returning character%s'%s
        test['f']="""\
      function f()
      character%s f
      f = "Y"
      end
"""%s
        test['py']="""\
