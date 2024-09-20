    from numpy.distutils.core import setup
    setup(configuration=configuration)
""" % (locals())
        setupfile = os.path.abspath('setup_extgen.py')
        f = open(setupfile, 'w')
        f.write(setup_py)
        f.close()
        setup_args = ['build_ext','--build-lib','.']
        setup_cmd = ' '.join([sys.executable,setupfile]+setup_args)
        build_dir = '.'
