import numpy as _sp

cdef object cont0_array(rk_state *state, rk_cont0 func, object size):
    cdef double *array_data
    cdef ndarray array "arrayObject"
    cdef long length
    cdef long i

    if size is None:
        return func(state)
    else:
        array = <ndarray>_sp.empty(size, _sp.Float64)
        length = PyArray_SIZE(array)
        array_data = <double *>array.data
        for i from 0 <= i < length:
            array_data[i] = func(state)
        return array

cdef object cont1_array(rk_state *state, rk_cont1 func, object size, double a):
    cdef double *array_data
    cdef ndarray array "arrayObject"
    cdef long length
    cdef long i

    if size is None:
        return func(state, a)
    else:
        array = <ndarray>_sp.empty(size, _sp.Float64)
        length = PyArray_SIZE(array)
        array_data = <double *>array.data
        for i from 0 <= i < length:
            array_data[i] = func(state, a)
        return array

cdef object cont2_array(rk_state *state, rk_cont2 func, object size, double a, 
    double b):
    cdef double *array_data
    cdef ndarray array "arrayObject"
    cdef long length
    cdef long i

    if size is None:
        return func(state, a, b)
    else:
        array = <ndarray>_sp.empty(size, _sp.Float64)
        length = PyArray_SIZE(array)
        array_data = <double *>array.data
        for i from 0 <= i < length:
            array_data[i] = func(state, a, b)
        return array

cdef object cont3_array(rk_state *state, rk_cont3 func, object size, double a, 
    double b, double c):

    cdef double *array_data
    cdef ndarray array "arrayObject"
    cdef long length
    cdef long i
    
    if size is None:
        return func(state, a, b, c)
    else:
        array = <ndarray>_sp.empty(size, _sp.Float64)
        length = PyArray_SIZE(array)
        array_data = <double *>array.data
        for i from 0 <= i < length:
            array_data[i] = func(state, a, b, c)
        return array

cdef object disc0_array(rk_state *state, rk_disc0 func, object size):
    cdef long *array_data
    cdef ndarray array "arrayObject"
    cdef long length
    cdef long i

    if size is None:
        return func(state)
    else:
        array = <ndarray>_sp.empty(size, _sp.Int)
        length = PyArray_SIZE(array)
        array_data = <long *>array.data
        for i from 0 <= i < length:
            array_data[i] = func(state)
        return array

cdef object discnp_array(rk_state *state, rk_discnp func, object size, long n, double p):
    cdef long *array_data
    cdef ndarray array "arrayObject"
    cdef long length
    cdef long i

    if size is None:
        return func(state, n, p)
    else:
        array = <ndarray>_sp.empty(size, _sp.Int)
        length = PyArray_SIZE(array)
        array_data = <long *>array.data
        for i from 0 <= i < length:
            array_data[i] = func(state, n, p)
        return array

cdef object discnmN_array(rk_state *state, rk_discnmN func, object size, 
    long n, long m, long N):
    cdef long *array_data
    cdef ndarray array "arrayObject"
    cdef long length
    cdef long i

    if size is None:
        return func(state, n, m, N)
    else:
        array = <ndarray>_sp.empty(size, _sp.Int)
        length = PyArray_SIZE(array)
        array_data = <long *>array.data
        for i from 0 <= i < length:
            array_data[i] = func(state, n, m, N)
        return array

cdef object discd_array(rk_state *state, rk_discd func, object size, double a):
    cdef long *array_data
    cdef ndarray array "arrayObject"
    cdef long length
    cdef long i

    if size is None:
        return func(state, a)
    else:
        array = <ndarray>_sp.empty(size, _sp.Int)
        length = PyArray_SIZE(array)
        array_data = <long *>array.data
        for i from 0 <= i < length:
            array_data[i] = func(state, a)
        return array

cdef double kahan_sum(double *darr, long n):
    cdef double c, y, t, sum
    cdef long i
    sum = darr[0]
    c = 0.0
    for i from 1 <= i < n:
        y = darr[i] - c
        t = sum + y
        c = (t-sum) - y
        sum = t
    return sum

cdef class RandomState:
    """Container for the Mersenne Twister PRNG.

    Constructor
    -----------
    RandomState(seed=None): initializes the PRNG with the given seed. See the
        seed() method for details.

    Distribution Methods
    -----------------
    RandomState exposes a number of methods for generating random numbers drawn
