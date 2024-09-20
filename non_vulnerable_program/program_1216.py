    from numpy.distutils.fcompiler import new_fcompiler
    #compiler = new_fcompiler(compiler='g95')
    compiler = G95FCompiler()
    compiler.customize()
    print compiler.get_version()



