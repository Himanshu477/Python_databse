    from numpy.distutils.fcompiler import new_fcompiler
    compiler = new_fcompiler(compiler='pathf95')
    compiler.customize()
    print compiler.get_version()


