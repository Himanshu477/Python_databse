from Numeric import *
import pprint,string
show=pprint.pprint
mmap={
    'real*8':{'pref':'d','in':4.7,'inout':5.3},
    'real':{'pref':'f','in':4,'inout':5.3},
    'integer*8':{'pref':'l','in':4,'inout':-7L},
    'integer':{'pref':'i','in':4.1,'inout':-7.2},
    'integer*2':{'pref':'s','in':4,'inout':-7},
    'integer*1':{'pref':'b','in':4,'inout':-7},
    'logical':{'pref':'i','in':0,'inout':1},
    'logical*1':{'pref':'1','in':0,'inout':1},
    'logical*2':{'pref':'s','in':0,'inout':1},
    'double precision':{'pref':'d','in':4.7,'inout':5.3},
    'complex':{'pref':'F','in':4.7-2j,'inout':5.3+4j},
    'double complex':{'pref':'D','in':4.7-2j,'inout':5.3+4j},
    'complex*16':{'pref':'D','in':4.7-2j,'inout':5.3+4j},
    }
#mmap={'double complex':{'pref':'D','in':4.7-2j,'inout':5.3+4j}}
for k in mmap.keys():
    if string.find(k,'logical')>=0:
        mmap[k]['inout_res']=not mmap[k]['inout']
        mmap[k]['out_res']=mmap[k]['in']
        mmap[k]['ret']=1
    else:
        mmap[k]['inout_res']=mmap[k]['inout']+mmap[k]['inout']
        mmap[k]['out_res']=mmap[k]['in']
        mmap[k]['ret']=mmap[k]['inout_res']+mmap[k]['in']
        if string.find(k,'integer')>=0:
            mmap[k]['inout_res'] = int(mmap[k]['inout_res'])
            mmap[k]['out_res'] = int(mmap[k]['out_res'])
            mmap[k]['ret'] = int(mmap[k]['ret'])
# Begin
ff=open(ffname,'w')
hf=open(hfname,'w')
py=open(pyname,'w')
hf.write('!%f90\npythonmodule iotest\n')
py.write("""\
