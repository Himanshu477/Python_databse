    import nose
    from numpy.testing.noseclasses import NumpyDoctest
    argv = ['', __file__, '--with-numpydoctest']
    nose.core.TestProgram(argv=argv, addplugins=[NumpyDoctest()])


