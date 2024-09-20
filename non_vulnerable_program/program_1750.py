    from numpy.distutils.core import setup
    setup(configuration=configuration)
''')
    f.close()
    print get_char_bit()
    os.system('python foo_setup.py config_fc --fcompiler=gnu95 build build_ext --inplace')
