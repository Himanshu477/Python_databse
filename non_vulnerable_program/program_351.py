    from fcompiler import new_fcompiler
    compiler = new_fcompiler(compiler='intel')
    compiler.customize()
    print compiler.get_version()


