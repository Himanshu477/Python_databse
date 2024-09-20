    from scipy.distutils.fcompiler import new_fcompiler
    compiler = new_fcompiler(compiler='lahey')
    compiler.customize()
    print compiler.get_version()


