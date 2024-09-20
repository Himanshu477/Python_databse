    from md5 import new as md5

#
# Maintaining a temporary module directory
#

_module_dir = None

def _cleanup():
    global _module_dir
    if _module_dir is not None:
        try:
            sys.path.remove(_module_dir)
        except ValueError:
            pass
        try:
            shutil.rmtree(_module_dir)
        except (IOError, OSError):
            pass
        _module_dir = None

def get_module_dir():
    global _module_dir
    if _module_dir is None:
        _module_dir = tempfile.mkdtemp()
        atexit.register(_cleanup)
        if _module_dir not in sys.path:
            sys.path.insert(0, _module_dir)
    return _module_dir

def get_temp_module_name():
    # Assume single-threaded, and the module dir usable only by this thread
    d = get_module_dir()
    for j in xrange(5403, 9999999):
        name = "_test_ext_module_%d" % j
        fn = os.path.join(d, name)
        if name not in sys.modules and not os.path.isfile(fn+'.py'):
            return name
    raise RuntimeError("Failed to create a temporary module name")

def _memoize(func):
    memo = {}
    def wrapper(*a, **kw):
        key = repr((a, kw))
        if key not in memo:
            try:
                memo[key] = func(*a, **kw)
            except Exception, e:
                memo[key] = e
                raise
        ret = memo[key]
        if isinstance(ret, Exception):
            raise ret
        return ret
    wrapper.__name__ = func.__name__
    return wrapper

#
# Building modules
#

@_memoize
def build_module(source_files, options=[], skip=[], only=[], module_name=None):
    """
    Compile and import a f2py module, built from the given files.

    """

    code = ("import sys; sys.path = %s; import numpy.f2py as f2py2e; "
            "f2py2e.main()" % repr(sys.path))

    d = get_module_dir()

    # Copy files
    dst_sources = []
    for fn in source_files:
        if not os.path.isfile(fn):
            raise RuntimeError("%s is not a file" % fn)
        dst = os.path.join(d, os.path.basename(fn))
        shutil.copyfile(fn, dst)
        dst_sources.append(dst)

    # Prepare options
    if module_name is None:
        module_name = get_temp_module_name()
    f2py_opts = ['-c', '-m', module_name] + options + dst_sources
    if skip:
        f2py_opts += ['skip:'] + skip
    if only:
        f2py_opts += ['only:'] + only

    # Build
    cwd = os.getcwd()
    try:
        os.chdir(d)
        cmd = [sys.executable, '-c', code] + f2py_opts
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        out, err = p.communicate()
        if p.returncode != 0:
            raise RuntimeError("Running f2py failed: %s\n%s"
                               % (cmd[4:], out))
    finally:
        os.chdir(cwd)

        # Partial cleanup
        for fn in dst_sources:
            os.unlink(fn)

    # Import
    __import__(module_name)
    return sys.modules[module_name]

@_memoize
def build_code(source_code, options=[], skip=[], only=[], suffix=None,
               module_name=None):
    """
    Compile and import Fortran code using f2py.

    """
    if suffix is None:
        suffix = '.f'

    fd, tmp_fn = tempfile.mkstemp(suffix=suffix)
    os.write(fd, asbytes(source_code))
    os.close(fd)

    try:
        return build_module([tmp_fn], options=options, skip=skip, only=only,
                            module_name=module_name)
    finally:
        os.unlink(tmp_fn)

#
# Check if compilers are available at all...
#

_compiler_status = None
def _get_compiler_status():
    global _compiler_status
    if _compiler_status is not None:
        return _compiler_status

    _compiler_status = (False, False, False)

    # XXX: this is really ugly. But I don't know how to invoke Distutils
    #      in a safer way...
    code = """
