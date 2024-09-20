    from numpy.distutils.core import setup
    setup(configuration=configuration)
""" % dict(config_code=config_code, syspath = repr(sys.path))

    script = os.path.join(d, get_temp_module_name() + '.py')
    dst_sources.append(script)
    f = open(script, 'wb')
    f.write(asbytes(code))
    f.close()

    # Build
    cwd = os.getcwd()
    try:
        os.chdir(d)
        cmd = [sys.executable, script, 'build_ext', '-i']
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        out, err = p.communicate()
        if p.returncode != 0:
            raise RuntimeError("Running distutils build failed: %s\n%s"
                               % (cmd[4:], out))
    finally:
        os.chdir(cwd)

        # Partial cleanup
        for fn in dst_sources:
            os.unlink(fn)

    # Import
    __import__(module_name)
    return sys.modules[module_name]

#
# Unittest convenience
#

class F2PyTest(object):
    code = None
    sources = None
    options = []
    skip = []
    only = []
    suffix = '.f'
    module = None
    module_name = None

    def setUp(self):
        if self.module is not None:
            return

        # Check compiler availability first
        if not has_c_compiler():
            raise nose.SkipTest("No C compiler available")

        codes = []
        if self.sources:
            codes.extend(self.sources)
        if self.code is not None:
            codes.append(self.suffix)

        needs_f77 = False
        needs_f90 = False
        for fn in codes:
            if fn.endswith('.f'):
                needs_f77 = True
            elif fn.endswith('.f90'):
                needs_f90 = True
        if needs_f77 and not has_f77_compiler():
            raise nose.SkipTest("No Fortran 77 compiler available")
        if needs_f90 and not has_f90_compiler():
            raise nose.SkipTest("No Fortran 90 compiler available")

        # Build the module
        if self.code is not None:
            self.module = build_code(self.code, options=self.options,
                                     skip=self.skip, only=self.only,
                                     suffix=self.suffix,
                                     module_name=self.module_name)

        if self.sources is not None:
            self.module = build_module(self.sources, options=self.options,
                                       skip=self.skip, only=self.only,
                                       module_name=self.module_name)


__all__ = ['savetxt', 'loadtxt', 'genfromtxt', 'ndfromtxt', 'mafromtxt',
        'recfromtxt', 'recfromcsv', 'load', 'loads', 'save', 'savez',
        'packbits', 'unpackbits', 'fromregex', 'DataSource']

