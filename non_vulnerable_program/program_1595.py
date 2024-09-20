    from numpy.distutils.fcompiler import new_fcompiler
    compiler = new_fcompiler(compiler='intel')
    compiler.customize()
    print compiler.get_version()


"""
Support code for building Python extensions on Windows.

    # NT stuff
    # 1. Make sure libpython<version>.a exists for gcc.  If not, build it.
    # 2. Force windows to use gcc (we're struggling with MSVC and g77 support)
    # 3. Force windows to use g77

"""

