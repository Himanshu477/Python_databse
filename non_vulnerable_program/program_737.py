        import f2py2e as f2py
        sys_argv = sys.argv
        mess(' f2py-ing..\n')
        pyf_sources = ['iotest.pyf','iotestrout.f']
        f2py_opts = []
        try:
            i = sys_argv.index('--no-wrap-functions')
        except ValueError:
            i = -1
        if i>=0:
            f2py_opts.append('no-wrap-functions')
            sys.argv = sys.argv[:i] + sys.argv[i+1:]

        if len(sys.argv)==1:
            sys.argv = sys.argv + ['build']
        if sys.argv[-1]=='build':
            sys.argv = sys.argv + ['--build-platlib','.']

        ############## building extension module ###########
