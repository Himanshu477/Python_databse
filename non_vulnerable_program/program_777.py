import f2pytest,sys
e = 1+2j
r = f2pytest.f()
if abs(r-e) > 1e-6:
    sys.stderr.write('expected %s but got %s\\n'%(e,r))
    sys.exit()
print 'ok'
"""
        tests.append(test)
################################################################
if 0 or all:
    for s in ['','*1','*2','*4']:
        test={}
        test['name']='Fortran function returning integer%s'%s
        test['f']="""\
      function f()
      integer%s f
      f = 2
      end
"""%s
        test['py']="""\
