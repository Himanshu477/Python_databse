import time

def recurse_compare(n):
    print 'Recursively computing the first %d fibonacci numbers:' % n
    t1 = time.time()
    for i in range(n):
        py_fib1(i)
    t2 = time.time()
    py = t2- t1
    print ' speed in python:', t2 - t1
    
    #load into cache
    c_fib1(i)
    t1 = time.time()
    for i in range(n):
        c_fib1(i)
    t2 = time.time()    
    print ' speed in c:',t2 - t1    
    print ' speed up: %3.2f' % (py/(t2-t1))

def loop_compare(m,n):
    print 'Loopin to compute the first %d fibonacci numbers:' % n
    t1 = time.time()
    for i in range(m):
        for i in range(n):
            py_fib2(i)
    t2 = time.time()
    py = (t2-t1)
    print ' speed in python:', (t2 - t1)/m
    
    #load into cache
    c_fib2(i)
    t1 = time.time()
    for i in range(m):
        for i in range(n):
            c_fib2(i)
    t2 = time.time()
    print ' speed in c:',(t2 - t1)/ m    
    print ' speed up: %3.2f' % (py/(t2-t1))
    
if __name__ == "__main__":
    n = 30
    recurse_compare(n)
    m= 1000    
    loop_compare(m,n)    
    print 'fib(30)', c_fib1(30),py_fib1(30),c_fib2(30),py_fib2(30)

