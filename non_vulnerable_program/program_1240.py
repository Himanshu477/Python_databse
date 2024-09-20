    from numpy.distutils.fcompiler import new_fcompiler
    compiler = new_fcompiler(compiler='mips')
    compiler.customize()
    print compiler.get_version()


