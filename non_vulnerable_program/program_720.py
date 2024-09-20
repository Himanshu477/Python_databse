import os,sys,re,time

def cmdline():
    m=re.compile(r'\A\d+\Z')
    args = []
    repeat = 1
    for a in sys.argv[1:]:
        if m.match(a):
            repeat = eval(a)
        else:
            args.append(a)
    f2py_opts = ' '.join(args)
    return repeat,f2py_opts

if sys.platform[:5]=='linux':
    def jiffies(_proc_pid_stat = '/proc/%s/stat'%(os.getpid()),
                _load_time=time.time()):
        """ Return number of jiffies (1/100ths of a second) that this
        process has been scheduled in user mode. See man 5 proc. """
        try:
            f=open(_proc_pid_stat,'r')
            l = f.readline().split(' ')
            f.close()
            return int(l[13])
        except:
            return int(100*(time.time()-_load_time))

    def memusage(_proc_pid_stat = '/proc/%s/stat'%(os.getpid())):
        """ Return virtual memory size in bytes of the running python.
        """
        try:
            f=open(_proc_pid_stat,'r')
            l = f.readline().split(' ')
            f.close()
            return int(l[22])
        except:
            return
else:
    def jiffies(_load_time=time.time()):
        """ Return number of jiffies (1/100ths of a second) that this
    process has been scheduled in user mode. [Emulation with time.time]. """
        return int(100*(time.time()-_load_time))
    
    def memusage():
        pass

def run(runtest,test_functions,repeat=1):
    l = [(t,repr(t.__doc__.split('\n')[1].strip())) for t in test_functions]
    #l = [(t,'') for t in test_functions]
    start_memusage = memusage()
    diff_memusage = None
    start_jiffies = jiffies()
    i = 0
    while i<repeat:
        i += 1
        for t,fname in l:
            runtest(t)
            if start_memusage is None: continue
            if diff_memusage is None:
                diff_memusage = memusage() - start_memusage
            else:
                diff_memusage2 = memusage() - start_memusage
                if diff_memusage2!=diff_memusage:
                    print 'memory usage change at step %i:' % i,\
                          diff_memusage2-diff_memusage,\
                          fname
                    diff_memusage = diff_memusage2
    current_memusage = memusage()
    print 'run',repeat*len(test_functions),'tests',\
          'in %.2f seconds' % ((jiffies()-start_jiffies)/100.0)
    if start_memusage:
        print 'initial virtual memory size:',start_memusage,'bytes'
        print 'current virtual memory size:',current_memusage,'bytes'


#!/usr/bin/env python
"""

Build F90 module support for f2py2e.

Copyright 2000 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@ioc.ee>          
Permission to use, modify, and distribute this software is given under the
terms of the LGPL.  See http://www.fsf.org

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2005/02/03 19:30:23 $
Pearu Peterson
"""

__version__ = "$Revision: 1.27 $"[10:-1]

f2py_version='See `f2py -v`'

