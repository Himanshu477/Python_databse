    from numpy.distutils.fcompiler import new_fcompiler
    #compiler = new_fcompiler(compiler='ibm')
    compiler = IbmFCompiler()
    compiler.customize()
    print compiler.get_version()


