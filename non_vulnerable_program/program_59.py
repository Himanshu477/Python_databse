import time


def compare(m,n):
    a = ones((n,n),Float64)
    type = Float32
    print 'Cast/Copy/Transposing (%d,%d)array %d times' % (n,n,m)
    t1 = time.time()
    for i in range(m):
        for i in range(n):
            b = _castCopyAndTranspose(type,a)
    t2 = time.time()
    py = (t2-t1)
    print ' speed in python:', (t2 - t1)/m
    

    # load into cache    
    b = cast_copy_transpose(type,a)
    t1 = time.time()
    for i in range(m):
        for i in range(n):
            b = cast_copy_transpose(type,a)
    t2 = time.time()
    print ' speed in c:',(t2 - t1)/ m    
    print ' speed up: %3.2f' % (py/(t2-t1))

    # inplace tranpose
    b = _inplace_transpose(a)
    t1 = time.time()
    for i in range(m):
        for i in range(n):
            b = _inplace_transpose(a)
    t2 = time.time()
    print ' inplace transpose c:',(t2 - t1)/ m    
    print ' speed up: %3.2f' % (py/(t2-t1))
    
if __name__ == "__main__":
    m,n = 1,150
    compare(m,n)    


# Borrowed from Alex Martelli's sort from Python cookbook using inlines
# 2x over fastest Python version -- again, maybe not worth the effort...
# Then again, 2x is 2x...
#
#    C:\home\ej\wrk\scipy\compiler\examples>python dict_sort.py
#    Dict sort of 1000 items for 300 iterations:
#     speed in python: 0.319999933243
#     [0, 1, 2, 3, 4]
#     speed in c: 0.159999966621
#     speed up: 2.00
#     [0, 1, 2, 3, 4]
 
