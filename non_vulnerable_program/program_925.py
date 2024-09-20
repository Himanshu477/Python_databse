    from scipy.distutils.fcompiler import new_fcompiler
    #compiler = new_fcompiler(compiler='gnu')
    compiler = GnuFCompiler()
    compiler.customize()
    print compiler.get_version()





