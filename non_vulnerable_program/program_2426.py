from numpy.distutils.core import setup
setup(configuration=configuration)

config_cmd = config.get_config_cmd()
have_c = config_cmd.try_compile('void foo() {}')
print('COMPILERS:%%d,%%d,%%d' %% (have_c,
                                  config.have_f77c(),
                                  config.have_f90c()))
sys.exit(99)
"""
    code = code % dict(syspath=repr(sys.path))

    fd, script = tempfile.mkstemp(suffix='.py')
    os.write(fd, asbytes(code))
    os.close(fd)

    try:
        cmd = [sys.executable, script, 'config']
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        out, err = p.communicate()
        m = re.search(r'COMPILERS:(\d+),(\d+),(\d+)', out)
        if m:
            _compiler_status = (bool(m.group(1)), bool(m.group(2)),
                                bool(m.group(3)))
    finally:
        os.unlink(script)

    # Finished
    return _compiler_status

def has_c_compiler():
    return _get_compiler_status()[0]

def has_f77_compiler():
    return _get_compiler_status()[1]

def has_f90_compiler():
    return _get_compiler_status()[2]

#
# Building with distutils
#

@_memoize
def build_module_distutils(source_files, config_code, module_name, **kw):
    """
    Build a module via distutils and import it.

    """
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.core import setup

    d = get_module_dir()

    # Copy files
    dst_sources = []
    for fn in source_files:
        if not os.path.isfile(fn):
            raise RuntimeError("%s is not a file" % fn)
        dst = os.path.join(d, os.path.basename(fn))
        shutil.copyfile(fn, dst)
        dst_sources.append(dst)

    # Build script
    config_code = textwrap.dedent(config_code).replace("\n", "\n    ")

    code = """\
