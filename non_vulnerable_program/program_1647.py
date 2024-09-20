    from numpy.distutils.fcompiler import new_fcompiler
    compiler = new_fcompiler(compiler='hpux')
    compiler.customize()
    print compiler.get_version()


