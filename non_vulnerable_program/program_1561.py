    from numpy.distutils.command import egg_info
    numpy_cmdclass['bdist_egg'] = bdist_egg.bdist_egg
    numpy_cmdclass['develop'] = develop.develop
    numpy_cmdclass['easy_install'] = easy_install.easy_install
    numpy_cmdclass['egg_info'] = egg_info.egg_info

def _dict_append(d, **kws):
    for k,v in kws.items():
        if not d.has_key(k):
            d[k] = v
            continue
        dv = d[k]
        if isinstance(dv, tuple):
            d[k] = dv + tuple(v)
        elif isinstance(dv, list):
            d[k] = dv + list(v)
        elif isinstance(dv, dict):
            _dict_append(dv, **v)
        elif is_string(dv):
            d[k] = dv + v
        else:
            raise TypeError,`type(dv)`

def _command_line_ok(_cache=[]):
    """ Return True if command line does not contain any
    help or display requests.
    """
    if _cache:
        return _cache[0]
    ok = True
    display_opts = ['--'+n for n in Distribution.display_option_names]
    for o in Distribution.display_options:
        if o[1]:
            display_opts.append('-'+o[1])
    for arg in sys.argv:
        if arg.startswith('--help') or arg=='-h' or arg in display_opts:
            ok = False
            break
    _cache.append(ok)
    return ok

def _exit_interactive_session(_cache=[]):
    if _cache:
        return # been here
    _cache.append(1)
    print '-'*72
    raw_input('Press ENTER to close the interactive session..')
    print '='*72

def get_distribution(always=False):
    dist = distutils.core._setup_distribution
    # XXX Hack to get numpy installable with easy_install.
    # The problem is easy_install runs it's own setup(), which
    # sets up distutils.core._setup_distribution. However,
    # when our setup() runs, that gets overwritten and lost.
    # We can't use isinstance, as the DistributionWithoutHelpCommands
    # class is local to a function in setuptools.command.easy_install
    if dist is not None and \
            repr(dist).find('DistributionWithoutHelpCommands') != -1:
        dist = None
    if always and dist is None:
        dist = distutils.dist.Distribution()
    return dist

def setup(**attr):

    if len(sys.argv)<=1:
        from interactive import interactive_sys_argv
        import atexit
        atexit.register(_exit_interactive_session)
        sys.argv[:] = interactive_sys_argv(sys.argv)
        if len(sys.argv)>1:
            return setup(**attr)

    cmdclass = numpy_cmdclass.copy()

    new_attr = attr.copy()
    if new_attr.has_key('cmdclass'):
        cmdclass.update(new_attr['cmdclass'])
    new_attr['cmdclass'] = cmdclass

    if new_attr.has_key('configuration'):
        # To avoid calling configuration if there are any errors
        # or help request in command in the line.
        configuration = new_attr.pop('configuration')

        old_dist = distutils.core._setup_distribution
        old_stop = distutils.core._setup_stop_after
        distutils.core._setup_distribution = None
        distutils.core._setup_stop_after = "commandline"
        try:
            dist = setup(**new_attr)
        finally:
            distutils.core._setup_distribution = old_dist
            distutils.core._setup_stop_after = old_stop
        if dist.help or not _command_line_ok():
            # probably displayed help, skip running any commands
            return dist

        # create setup dictionary and append to new_attr
        config = configuration()
        if hasattr(config,'todict'):
            config = config.todict()
        _dict_append(new_attr, **config)

    # Move extension source libraries to libraries
    libraries = []
    for ext in new_attr.get('ext_modules',[]):
        new_libraries = []
        for item in ext.libraries:
            if is_sequence(item):
                lib_name, build_info = item
                _check_append_ext_library(libraries, item)
                new_libraries.append(lib_name)
            elif is_string(item):
                new_libraries.append(item)
            else:
                raise TypeError("invalid description of extension module "
                                "library %r" % (item,))
        ext.libraries = new_libraries
    if libraries:
        if not new_attr.has_key('libraries'):
            new_attr['libraries'] = []
        for item in libraries:
            _check_append_library(new_attr['libraries'], item)

    # sources in ext_modules or libraries may contain header files
    if (new_attr.has_key('ext_modules') or new_attr.has_key('libraries')) \
       and not new_attr.has_key('headers'):
        new_attr['headers'] = []

    return old_setup(**new_attr)

def _check_append_library(libraries, item):
    for libitem in libraries:
        if is_sequence(libitem):
            if is_sequence(item):
                if item[0]==libitem[0]:
                    if item[1] is libitem[1]:
                        return
                    warnings.warn("[0] libraries list contains %r with"
                                  " different build_info" % (item[0],))
                    break
            else:
                if item==libitem[0]:
                    warnings.warn("[1] libraries list contains %r with"
                                  " no build_info" % (item[0],))
                    break
        else:
            if is_sequence(item):
                if item[0]==libitem:
                    warnings.warn("[2] libraries list contains %r with"
                                  " no build_info" % (item[0],))
                    break
            else:
                if item==libitem:
                    return
    libraries.append(item)

def _check_append_ext_library(libraries, (lib_name,build_info)):
    for item in libraries:
        if is_sequence(item):
            if item[0]==lib_name:
                if item[1] is build_info:
                    return
                warnings.warn("[3] libraries list contains %r with"
                              " different build_info" % (lib_name,))
                break
        elif item==lib_name:
            warnings.warn("[4] libraries list contains %r with"
                          " no build_info" % (lib_name,))
            break
    libraries.append((lib_name,build_info))


#!/usr/bin/python
"""

process_file(filename)

  takes templated file .xxx.src and produces .xxx file where .xxx
  is .pyf .f90 or .f using the following template rules:

  '<..>' denotes a template.

  All function and subroutine blocks in a source file with names that
  contain '<..>' will be replicated according to the rules in '<..>'.

  The number of comma-separeted words in '<..>' will determine the number of
  replicates.

  '<..>' may have two different forms, named and short. For example,

  named:
   <p=d,s,z,c> where anywhere inside a block '<p>' will be replaced with
   'd', 's', 'z', and 'c' for each replicate of the block.

   <_c>  is already defined: <_c=s,d,c,z>
   <_t>  is already defined: <_t=real,double precision,complex,double complex>

  short:
   <s,d,c,z>, a short form of the named, useful when no <p> appears inside
   a block.

  In general, '<..>' contains a comma separated list of arbitrary
  expressions. If these expression must contain a comma|leftarrow|rightarrow,
  then prepend the comma|leftarrow|rightarrow with a backslash.

  If an expression matches '\\<index>' then it will be replaced
  by <index>-th expression.

  Note that all '<..>' forms in a block must have the same number of
  comma-separated entries.

 Predefined named template rules:
  <prefix=s,d,c,z>
  <ftype=real,double precision,complex,double complex>
  <ftypereal=real,double precision,\\0,\\1>
  <ctype=float,double,complex_float,complex_double>
  <ctypereal=float,double,\\0,\\1>

"""

__all__ = ['process_str','process_file']

