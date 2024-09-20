import f2pytest,sys
def g():
    return 34.56
r = f2pytest.f(g)
if abs(r-34.56)>1e-5:
    sys.stderr.write('expected 34.56 but got %s\\n'%r)
    sys.exit()
print 'ok'
"""
        tests.append(test)
#################################################################
if 0 or all:
    for s in ['','*8','*16','*32']:
        if s=='*32' and skip: continue
        test={}
        test['trydefine']=['-DF2PY_CB_RETURNCOMPLEX']
        test['name']='Callback function returning complex%s'%s
        test['f']="""\
      function f(g)
      complex%s f,g
      external g
      f=g()
      end
"""%s
        test['py']="""\
