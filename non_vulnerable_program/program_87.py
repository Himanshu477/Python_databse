import time
import RandomArray
def compare(m,Nobs,Ncodes,Nfeatures):
    obs = RandomArray.normal(0.,1.,(Nobs,Nfeatures))
    codes = RandomArray.normal(0.,1.,(Ncodes,Nfeatures))
    import scipy.cluster.vq
    scipy.cluster.vq
    print 'vq with %d observation, %d features and %d codes for %d iterations' % \
           (Nobs,Nfeatures,Ncodes,m)
    t1 = time.time()
    for i in range(m):
        code,dist = scipy.cluster.vq.py_vq(obs,codes)
    t2 = time.time()
    py = (t2-t1)
    print ' speed in python:', (t2 - t1)/m
    print code[:2],dist[:2]
    
    t1 = time.time()
    for i in range(m):
        code,dist = scipy.cluster.vq.vq(obs,codes)
    t2 = time.time()
    print ' speed in standard c:', (t2 - t1)/m
    print code[:2],dist[:2]
    print ' speed up: %3.2f' % (py/(t2-t1))
    
    # load into cache    
    b = vq(obs,codes)
    t1 = time.time()
    for i in range(m):
        code,dist = vq(obs,codes)
    t2 = time.time()
    print ' speed inline/blitz:',(t2 - t1)/ m    
    print code[:2],dist[:2]
    print ' speed up: %3.2f' % (py/(t2-t1))

    # load into cache    
    b = vq2(obs,codes)
    t1 = time.time()
    for i in range(m):
        code,dist = vq2(obs,codes)
    t2 = time.time()
    print ' speed inline/blitz2:',(t2 - t1)/ m    
    print code[:2],dist[:2]
    print ' speed up: %3.2f' % (py/(t2-t1))

    # load into cache    
    b = vq3(obs,codes)
    t1 = time.time()
    for i in range(m):
        code,dist = vq3(obs,codes)
    t2 = time.time()
    print ' speed using C arrays:',(t2 - t1)/ m    
    print code[:2],dist[:2]
    print ' speed up: %3.2f' % (py/(t2-t1))
    
if __name__ == "__main__":
    compare(100,1000,30,10)    
    #compare(1,10,2,10)    


""" This is taken from the scrolled window example from the demo.

    Take a look at the DoDrawing2() method below.  The first 6 lines
    or so have been translated into C++.
    
"""


