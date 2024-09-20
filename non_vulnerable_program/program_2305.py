    import Instant, numpy
    ext = Instant.Instant()
    ext.create_extension(code=s, headers=["numpy/arrayobject.h"],
                         include_dirs=[numpy.get_include()],
                         init_code='import_array();', module="test2b_ext")
