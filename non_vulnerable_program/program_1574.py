    import numpy.numarray.util as nnu
    self.add_include_dirs(*nnu.get_numarray_include_dirs())
    
    def add_headers(self,*files):
        """ Add installable headers to configuration.
        Argument(s) can be either
        - 2-sequence (<includedir suffix>,<path to header file(s)>)
        - path(s) to header file(s) where python includedir suffix will default
          to package name.
        """
        headers = []
        for path in files:
            if is_string(path):
                [headers.append((self.name,p)) for p in self.paths(path)]
            else:
                if not isinstance(path, (tuple, list)) or len(path) != 2:
                    raise TypeError(repr(path))
                [headers.append((path[0],p)) for p in self.paths(path[1])]
        dist = self.get_distribution()
        if dist is not None:
            dist.headers.extend(headers)
        else:
            self.headers.extend(headers)
        return

    def paths(self,*paths,**kws):
        """ Apply glob to paths and prepend local_path if needed.
        """
        include_non_existing = kws.get('include_non_existing',True)
        return gpaths(paths,
                      local_path = self.local_path,
                      include_non_existing=include_non_existing)

    def _fix_paths_dict(self,kw):
        for k in kw.keys():
            v = kw[k]
            if k in ['sources','depends','include_dirs','library_dirs',
                     'module_dirs','extra_objects']:
                new_v = self.paths(v)
                kw[k] = new_v
        return

    def add_extension(self,name,sources,**kw):
        """ Add extension to configuration.

        Keywords:
          include_dirs, define_macros, undef_macros,
          library_dirs, libraries, runtime_library_dirs,
          extra_objects, extra_compile_args, extra_link_args,
          export_symbols, swig_opts, depends, language,
          f2py_options, module_dirs
          extra_info - dict or list of dict of keywords to be
                       appended to keywords.
        """
        ext_args = copy.copy(kw)
        ext_args['name'] = dot_join(self.name,name)
        ext_args['sources'] = sources

        if ext_args.has_key('extra_info'):
            extra_info = ext_args['extra_info']
            del ext_args['extra_info']
            if isinstance(extra_info, dict):
                extra_info = [extra_info]
            for info in extra_info:
                assert isinstance(info, dict), repr(info)
                dict_append(ext_args,**info)

        self._fix_paths_dict(ext_args)

        # Resolve out-of-tree dependencies
        libraries = ext_args.get('libraries',[])
        libnames = []
        ext_args['libraries'] = []
        for libname in libraries:
            if isinstance(libname,tuple):
                self._fix_paths_dict(libname[1])

            # Handle library names of the form libname@relative/path/to/library
            if '@' in libname:
                lname,lpath = libname.split('@',1)
                lpath = os.path.abspath(njoin(self.local_path,lpath))
                if os.path.isdir(lpath):
                    c = self.get_subpackage(None,lpath,
                                            caller_level = 2)
                    if isinstance(c,Configuration):
                        c = c.todict()
                    for l in [l[0] for l in c.get('libraries',[])]:
                        llname = l.split('__OF__',1)[0]
                        if llname == lname:
                            c.pop('name',None)
                            dict_append(ext_args,**c)
                            break
                    continue
            libnames.append(libname)

        ext_args['libraries'] = libnames + ext_args['libraries']

        from numpy.distutils.core import Extension
        ext = Extension(**ext_args)
        self.ext_modules.append(ext)

        dist = self.get_distribution()
        if dist is not None:
            self.warn('distutils distribution has been initialized,'\
                      ' it may be too late to add an extension '+name)
        return ext

    def add_library(self,name,sources,**build_info):
        """ Add library to configuration.

        Valid keywords for build_info:
          depends
          macros
          include_dirs
          extra_compiler_args
          f2py_options
        """
        build_info = copy.copy(build_info)
        name = name #+ '__OF__' + self.name
        build_info['sources'] = sources

        self._fix_paths_dict(build_info)

        self.libraries.append((name,build_info))

        dist = self.get_distribution()
        if dist is not None:
            self.warn('distutils distribution has been initialized,'\
                      ' it may be too late to add a library '+ name)
        return

    def add_scripts(self,*files):
        """ Add scripts to configuration.
        """
        scripts = self.paths(files)
        dist = self.get_distribution()
        if dist is not None:
            dist.scripts.extend(scripts)
        else:
            self.scripts.extend(scripts)
        return

    def dict_append(self,**dict):
        for key in self.list_keys:
            a = getattr(self,key)
            a.extend(dict.get(key,[]))
        for key in self.dict_keys:
            a = getattr(self,key)
            a.update(dict.get(key,{}))
        known_keys = self.list_keys + self.dict_keys + self.extra_keys
        for key in dict.keys():
            if key not in known_keys:
                a = getattr(self, key, None)
                if a and a==dict[key]: continue
                self.warn('Inheriting attribute %r=%r from %r' \
                          % (key,dict[key],dict.get('name','?')))
                setattr(self,key,dict[key])
                self.extra_keys.append(key)
            elif key in self.extra_keys:
                self.info('Ignoring attempt to set %r (from %r to %r)' \
                          % (key, getattr(self,key), dict[key]))
            elif key in known_keys:
                # key is already processed above
                pass
            else:
                raise ValueError, "Don't know about key=%r" % (key)
        return

    def __str__(self):
        from pprint import pformat
        known_keys = self.list_keys + self.dict_keys + self.extra_keys
        s = '<'+5*'-' + '\n'
        s += 'Configuration of '+self.name+':\n'
        known_keys.sort()
        for k in known_keys:
            a = getattr(self,k,None)
            if a:
                s += '%s = %s\n' % (k,pformat(a))
        s += 5*'-' + '>'
        return s

    def get_config_cmd(self):
        cmd = get_cmd('config')
        cmd.ensure_finalized()
        cmd.dump_source = 0
        cmd.noisy = 0
        old_path = os.environ.get('PATH')
        if old_path:
            path = os.pathsep.join(['.',old_path])
            os.environ['PATH'] = path
        return cmd

    def get_build_temp_dir(self):
        cmd = get_cmd('build')
        cmd.ensure_finalized()
        return cmd.build_temp

    def have_f77c(self):
        """ Check for availability of Fortran 77 compiler.
        Use it inside source generating function to ensure that
        setup distribution instance has been initialized.
        """
        simple_fortran_subroutine = '''
        subroutine simple
        end
        '''
        config_cmd = self.get_config_cmd()
        flag = config_cmd.try_compile(simple_fortran_subroutine,lang='f77')
        return flag

    def have_f90c(self):
        """ Check for availability of Fortran 90 compiler.
        Use it inside source generating function to ensure that
        setup distribution instance has been initialized.
        """
        simple_fortran_subroutine = '''
        subroutine simple
        end
        '''
        config_cmd = self.get_config_cmd()
        flag = config_cmd.try_compile(simple_fortran_subroutine,lang='f90')
        return flag

    def append_to(self, extlib):
        """ Append libraries, include_dirs to extension or library item.
        """
        if is_sequence(extlib):
            lib_name, build_info = extlib
            dict_append(build_info,
                        libraries=self.libraries,
                        include_dirs=self.include_dirs)
        else:
            from numpy.distutils.core import Extension
            assert isinstance(extlib,Extension), repr(extlib)
            extlib.libraries.extend(self.libraries)
            extlib.include_dirs.extend(self.include_dirs)
        return

    def _get_svn_revision(self,path):
        """ Return path's SVN revision number.
        """
        entries = njoin(path,'.svn','entries')
        revision = None
        if os.path.isfile(entries):
            f = open(entries)
            m = re.search(r'revision="(?P<revision>\d+)"',f.read())
            f.close()
            if m:
                revision = int(m.group('revision'))
        return revision

    def get_version(self, version_file=None, version_variable=None):
        """ Try to get version string of a package.
        """
        version = getattr(self,'version',None)
        if version is not None:
            return version

        # Get version from version file.
        if version_file is None:
            files = ['__version__.py',
                     self.name.split('.')[-1]+'_version.py',
                     'version.py',
                     '__svn_version__.py']
        else:
            files = [version_file]
        if version_variable is None:
            version_vars = ['version',
                            '__version__',
                            self.name.split('.')[-1]+'_version']
        else:
            version_vars = [version_variable]
        for f in files:
            fn = njoin(self.local_path,f)
            if os.path.isfile(fn):
                info = (open(fn),fn,('.py','U',1))
                name = os.path.splitext(os.path.basename(fn))[0]
                n = dot_join(self.name,name)
                try:
                    version_module = imp.load_module('_'.join(n.split('.')),*info)
                except ImportError,msg:
                    self.warn(str(msg))
                    version_module = None
                if version_module is None:
                    continue

                for a in version_vars:
                    version = getattr(version_module,a,None)
                    if version is not None:
                        break
                if version is not None:
                    break

        if version is not None:
            self.version = version
            return version

        # Get version as SVN revision number
        revision = self._get_svn_revision(self.local_path)
        if revision is not None:
            version = str(revision)
            self.version = version

        return version

    def make_svn_version_py(self):
        """ Generate package __svn_version__.py file from SVN revision number,
        it will be removed after python exits but will be available
        when sdist, etc commands are executed.

        If __svn_version__.py existed before, nothing is done.
        """
        target = njoin(self.local_path,'__svn_version__.py')
        if os.path.isfile(target):
            return
        def generate_svn_version_py():
            if not os.path.isfile(target):
                revision = self._get_svn_revision(self.local_path)
                assert revision is not None,'hmm, why I am not inside SVN tree???'
                version = str(revision)
                self.info('Creating %s (version=%r)' % (target,version))
                f = open(target,'w')
                f.write('version = %r\n' % (version))
                f.close()

            import atexit
            def rm_file(f=target,p=self.info):
                try: os.remove(f); p('removed '+f)
                except OSError: pass
                try: os.remove(f+'c'); p('removed '+f+'c')
                except OSError: pass
            atexit.register(rm_file)

            return target

        self.add_data_files(('', generate_svn_version_py()))

    def make_config_py(self,name='__config__'):
        """ Generate package __config__.py file containing system_info
        information used during building the package.
        """
        self.py_modules.append((self.name,name,generate_config_py))
        return

    def get_info(self,*names):
        """ Get resources information.
        """
        from system_info import get_info, dict_append
        info_dict = {}
        for a in names:
            dict_append(info_dict,**get_info(a))
        return info_dict


def get_cmd(cmdname, _cache={}):
    if not _cache.has_key(cmdname):
        import distutils.core
        dist = distutils.core._setup_distribution
        if dist is None:
            from distutils.errors import DistutilsInternalError
            raise DistutilsInternalError(
                  'setup distribution instance not initialized')
        cmd = dist.get_command_obj(cmdname)
        _cache[cmdname] = cmd
    return _cache[cmdname]

def get_numpy_include_dirs():
    # numpy_include_dirs are set by numpy/core/setup.py, otherwise []
    include_dirs = Configuration.numpy_include_dirs[:]
    if not include_dirs:
        import numpy
        if numpy.show_config is None:
            # running from numpy_core source directory
            include_dirs.append(njoin(os.path.dirname(numpy.__file__),
                                             'core', 'include'))
        else:
            # using installed numpy core headers
            import numpy.core as core
            include_dirs.append(njoin(os.path.dirname(core.__file__), 'include'))
    # else running numpy/core/setup.py
    return include_dirs

#########################

def default_config_dict(name = None, parent_name = None, local_path=None):
    """ Return a configuration dictionary for usage in
    configuration() function defined in file setup_<name>.py.
    """
    import warnings
    warnings.warn('Use Configuration(%r,%r,top_path=%r) instead of '\
                  'deprecated default_config_dict(%r,%r,%r)'
                  % (name, parent_name, local_path,
                     name, parent_name, local_path,
                     ))
    c = Configuration(name, parent_name, local_path)
    return c.todict()


def dict_append(d, **kws):
    for k, v in kws.items():
        if d.has_key(k):
            d[k].extend(v)
        else:
            d[k] = v

def appendpath(prefix, path):
    if os.path.sep != '/':
        prefix = prefix.replace('/', os.path.sep)
        path = path.replace('/', os.path.sep)
    drive = ''
    if os.path.isabs(path):
        drive = os.path.splitdrive(prefix)[0]
        absprefix = os.path.splitdrive(os.path.abspath(prefix))[1]
        pathdrive, path = os.path.splitdrive(path)
        d = os.path.commonprefix([absprefix, path])
        if os.path.join(absprefix[:len(d)], absprefix[len(d):]) != absprefix \
           or os.path.join(path[:len(d)], path[len(d):]) != path:
            # Handle invalid paths
            d = os.path.dirname(d)
        subpath = path[len(d):]
        if os.path.isabs(subpath):
            subpath = subpath[1:]
    else:
        subpath = path
    return os.path.normpath(njoin(drive + prefix, subpath))

def generate_config_py(target):
    """ Generate config.py file containing system_info information
    used during building the package.

    Usage:\
        config['py_modules'].append((packagename, '__config__',generate_config_py))
    """
    from numpy.distutils.system_info import system_info
    from distutils.dir_util import mkpath
    mkpath(os.path.dirname(target))
    f = open(target, 'w')
    f.write('# This file is generated by %s\n' % (os.path.abspath(sys.argv[0])))
    f.write('# It contains system_info results at the time of building this package.\n')
    f.write('__all__ = ["get_info","show"]\n\n')
    for k, i in system_info.saved_results.items():
        f.write('%s=%r\n' % (k, i))
    f.write('\ndef get_info(name):\n    g=globals()\n    return g.get(name,g.get(name+"_info",{}))\n')
    f.write('''
def show():
    for name,info_dict in globals().items():
        if name[0]=="_" or type(info_dict) is not type({}): continue
        print name+":"
        if not info_dict:
            print "  NOT AVAILABLE"
        for k,v in info_dict.items():
            v = str(v)
            if k==\'sources\' and len(v)>200: v = v[:60]+\' ...\\n... \'+v[-60:]
            print \'    %s = %s\'%(k,v)
        print
    return
    ''')

    f.close()
    return target


__all__ = ['ones','zeros','empty','identity','rand','randn','eye']

