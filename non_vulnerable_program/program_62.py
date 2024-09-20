import time

def sort_compare(a,n):
    print 'Dict sort of %d items for %d iterations:'%(len(a),n)
    t1 = time.time()
    for i in range(n):
        b=sortedDictValues3(a)
    t2 = time.time()
    py = (t2-t1)
    print ' speed in python:', (t2 - t1)
    print b[:5]
    
    b=c_sort(a)
    t1 = time.time()
    for i in range(n):
        b=c_sort(a)
    t2 = time.time()
    print ' speed in c:',(t2 - t1)    
    print ' speed up: %3.2f' % (py/(t2-t1))
    print b[:5]
def setup_dict(m):
    " does insertion order matter?"
    import whrandom
    a = range(m)
    d = {}
    for i in range(m):
        key = whrandom.choice(a)
        a.remove(key)
        d[key]=key
    return d    
if __name__ == "__main__":
    m = 1000
    a = setup_dict(m)
    n = 300
    sort_compare(a,n)    


# Typical run:
# C:\home\ej\wrk\scipy\compiler\examples>python fibonacci.py
# Recursively computing the first 30 fibonacci numbers:
#  speed in python: 3.98599994183
#  speed in c: 0.0800000429153
#  speed up: 49.82
# Loopin to compute the first 30 fibonacci numbers:
#  speed in python: 0.00053100001812
#  speed in c: 5.99999427795e-005
#  speed up: 8.85
# fib(30) 832040 832040 832040 832040

