import os
import sys
path = sys.path

for kind in ['f2py']:#['ctypes', 'pyrex', 'weave', 'f2py']:
        res[kind] = []
        sys.path = ['/Users/oliphant/numpybook/%s' % (kind,)] + path
        print sys.path
        for n in N:
            print "%s - %d" % (kind, n)
            t = timeit.Timer(eval('%s_run'%kind), eval('%s_pre %% (%d,%d)'%(kind,n,n)))
            mytime = min(t.repeat(3,100))
            res[kind].append(mytime)




