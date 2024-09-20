    from fcompiler import new_fcompiler
    compiler = new_fcompiler(compiler='nag')
    compiler.customize()
    print compiler.get_version()


