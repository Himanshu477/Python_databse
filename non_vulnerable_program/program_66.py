    import fibonacci_ext
c_fib1 = fibonacci_ext.c_fib1
c_fib2 = fibonacci_ext.c_fib2

#################################################################
# This where it might normally end, but we've added some timings
# below. Recursive solutions are much slower, and C is 10-50x faster
# than equivalent in Python for this simple little routine
#
#################################################################

def py_fib1(a):
    if a <= 2:
        return 1
    else:
        return py_fib1(a-2) + py_fib1(a-1)

def py_fib2(a):
    if a <= 2:
        return 1            
    last = next_to_last = 1
    for i in range(2,a):
        result = last + next_to_last
        next_to_last = last
        last = result
    return result;

