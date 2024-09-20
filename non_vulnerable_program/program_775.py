import f2pytest,sys
e = 'abcdefgh'
r = f2pytest.f()
if not e==r:
    sys.stderr.write('expected %s but got %s (ok)\\n'%(`e`,`r`))
    sys.exit()
print 'ok'
"""
    tests.append(test)
################################################################
if 0 or all:
    for s in ['','*4','*8','*16']:
        if s=='*16' and skip: continue
        test={}
        #test['f2pyflags']=['--debug-capi']
        test['name']='Fortran function returning real%s'%s
        test['f']="""\
      function f()
      real%s f
      f = 13.45
      end
"""%s
        test['py']="""\
