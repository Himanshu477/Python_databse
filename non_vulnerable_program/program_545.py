    from scipy.distutils.fcompiler import new_fcompiler
    compiler = new_fcompiler(compiler='ibm')
    compiler.customize()
    print compiler.get_version()


# http://developer.intel.com/software/products/compilers/flin/

