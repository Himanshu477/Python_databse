    import_array()

    def zadd(object ao, object bo):
        cdef ndarray c, a, b
        cdef npy_intp i
        a = c_numpy.PyArray_ContiguousFromAny(ao,
                      NPY_CDOUBLE, 1, 1)
        b = c_numpy.PyArray_ContiguousFromAny(bo,
                      NPY_CDOUBLE, 1, 1)
        c = c_numpy.PyArray_SimpleNew(a.nd, a.dimensions,
                     a.descr.type_num)
        for i from 0 <= i < a.dimensions[0]:
            (<npy_cdouble *>c.data)[i].real = \
                 (<npy_cdouble *>a.data)[i].real + \
                 (<npy_cdouble *>b.data)[i].real
            (<npy_cdouble *>c.data)[i].imag = \
                 (<npy_cdouble *>a.data)[i].imag + \
                 (<npy_cdouble *>b.data)[i].imag
        return c

This module shows use of the ``cimport`` statement to load the
definitions from the c_numpy.pxd file. As shown, both versions of the
