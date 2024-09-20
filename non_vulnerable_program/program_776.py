import f2pytest,sys
e = 13.45
r = f2pytest.f()
if abs(r-e) > 1e-6:
    sys.stderr.write('expected %s but got %s\\n'%(e,r))
    sys.exit()
print 'ok'
"""
        tests.append(test)
        break
################################################################
if 0 or all:
    for s in ['','*8','*16','*32']:
        if s=='*32' and skip: continue
        test={}
        #test['f2pyflags']=['--debug-capi']
        test['name']='Fortran function returning complex%s'%s
        test['f']="""\
      function f()
      complex%s f
      f = (1,2)
      end
"""%s
        test['py']="""\
