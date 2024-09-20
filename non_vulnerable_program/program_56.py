import time

def search_compare(a,n):
    print 'Binary search for %d items in %d length list of integers:'%(n,m)
    t1 = time.time()
    for i in range(n):
        py_int_search(a,i)
    t2 = time.time()
    py = (t2-t1)
    print ' speed in python:', (t2 - t1)

    # bisect
    t1 = time.time()
    for i in range(n):
        bisect(a,i)
    t2 = time.time()
    bi = (t2-t1)
    print ' speed of bisect:', bi
    print ' speed up: %3.2f' % (py/bi)

    # get it in cache
    c_int_search(a,i)
    t1 = time.time()
    for i in range(n):
        c_int_search(a,i,chk=1)
    t2 = time.time()
    print ' speed in c:',(t2 - t1)    
    print ' speed up: %3.2f' % (py/(t2-t1))

    # get it in cache
    c_int_search(a,i)
    t1 = time.time()
    for i in range(n):
        c_int_search(a,i,chk=0)
    t2 = time.time()
    print ' speed in c(no asserts):',(t2 - t1)    
    print ' speed up: %3.2f' % (py/(t2-t1))

    # get it in cache
    try:
        a = array(a)
        c_array_int_search(a,i)
        t1 = time.time()
        for i in range(n):
            c_array_int_search(a,i)
        t2 = time.time()
        print ' speed in c(Numeric arrays):',(t2 - t1)    
        print ' speed up: %3.2f' % (py/(t2-t1))
    except:
        pass
        
if __name__ == "__main__":
    # note bisect returns index+1 compared to other algorithms
    m= 100000
    a = range(m)
    n = 50000
    search_compare(a,n)    
    print 'search(a,3450)', c_int_search(a,3450), py_int_search(a,3450), bisect(a,3450)
    print 'search(a,-1)', c_int_search(a,-1), py_int_search(a,-1), bisect(a,-1)
    print 'search(a,10001)', c_int_search(a,10001), py_int_search(a,10001),bisect(a,10001)

""" Cast Copy Tranpose is used in Numeric's LinearAlgebra.py to convert
    C ordered arrays to Fortran order arrays before calling Fortran
    functions.  A couple of C implementations are provided here that 
    show modest speed improvements.  One is an "inplace" transpose that
    does an in memory transpose of an arrays elements.  This is the
    fastest approach and is beneficial if you don't need to keep the
    original array.    
"""
# C:\home\ej\wrk\scipy\compiler\examples>python cast_copy_transpose.py
# Cast/Copy/Transposing (150,150)array 1 times
#  speed in python: 0.870999932289
#  speed in c: 0.25
#  speed up: 3.48
#  inplace transpose c: 0.129999995232
#  speed up: 6.70

