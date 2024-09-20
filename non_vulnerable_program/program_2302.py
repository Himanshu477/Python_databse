    import_array()
    def filter(object ao):
        cdef ndarray a, b
        cdef npy_intp i, j, M, N, oS
        cdef npy_intp r,rm1,rp1,c,cm1,cp1
        cdef double value
        # Require an ALIGNED array
        # (but not necessarily contiguous)
        #  We will use strides to access the elements.
        a = c_numpy.PyArray_FROMANY(ao, NPY_DOUBLE, \
                    2, 2, NPY_ALIGNED)
        b = c_numpy.PyArray_SimpleNew(a.nd,a.dimensions, \
                                      a.descr.type_num)
        M = a.dimensions[0]
        N = a.dimensions[1]
        S0 = a.strides[0]
        S1 = a.strides[1]
        for i from 1 <= i < M-1:
            r = i*S0
            rm1 = r-S0
            rp1 = r+S0
            oS = i*N
            for j from 1 <= j < N-1:
                c = j*S1
                cm1 = c-S1
                cp1 = c+S1
                (<double *>b.data)[oS+j] = \
                   (<double *>(a.data+r+c))[0] + \
                   ((<double *>(a.data+rm1+c))[0] + \
                    (<double *>(a.data+rp1+c))[0] + \
                    (<double *>(a.data+r+cm1))[0] + \
                    (<double *>(a.data+r+cp1))[0])*0.5 + \
                   ((<double *>(a.data+rm1+cm1))[0] + \
                    (<double *>(a.data+rp1+cm1))[0] + \
                    (<double *>(a.data+rp1+cp1))[0] + \
                    (<double *>(a.data+rm1+cp1))[0])*0.25
        return b

This 2-d averaging filter runs quickly because the loop is in C and
the pointer computations are done only as needed. However, it is not
particularly easy to understand what is happening. A 2-d image, ``in``
, can be filtered using this code very quickly using:

.. code-block:: python

