    from __core_config__ import show as show_core_config
except ImportError:
    show_core_config = None

class PackageLoader:
    def __init__(self):
        """ Manages loading SciPy packages.
        """

        self.parent_frame = frame = sys._getframe(1)
        self.parent_name = eval('__name__',frame.f_globals,frame.f_locals)
        self.parent_path = eval('__path__',frame.f_globals,frame.f_locals)
        if not frame.f_locals.has_key('__all__'):
            exec('__all__ = []',frame.f_globals,frame.f_locals)
        self.parent_export_names = eval('__all__',frame.f_globals,frame.f_locals)

        self.info_modules = None
        self.imported_packages = []
        self.verbose = None

    def _get_info_files(self, package_dir, parent_path, parent_package=None):
        """ Return list of (package name,info.py file) from parent_path subdirectories.
        """
        from glob import glob
        files = glob(os.path.join(parent_path,package_dir,'info.py'))
        for info_file in glob(os.path.join(parent_path,package_dir,'info.pyc')):
            if info_file[:-1] not in files:
                files.append(info_file)
        info_files = []
        for info_file in files:
            package_name = os.path.dirname(info_file[len(parent_path)+1:])\
                           .replace(os.sep,'.')
            if parent_package:
                package_name = parent_package + '.' + package_name
            info_files.append((package_name,info_file))
            info_files.extend(self._get_info_files('*',
                                                   os.path.dirname(info_file),
                                                   package_name))
        return info_files

    def _init_info_modules(self, packages=None):
        """Initialize info_modules = {<package_name>: <package info.py module>}.
        """
        import imp
        info_files = []
        if packages is None:
            for path in self.parent_path:
                info_files.extend(self._get_info_files('*',path))
        else:
            for package_name in packages:
                package_dir = os.path.join(*package_name.split('.'))
                for path in self.parent_path:
                    names_files = self._get_info_files(package_dir, path)
                    if names_files:
                        info_files.extend(names_files)
                        break
                else:
                    self.warn('Package %r does not have info.py file. Ignoring.'\
                              % package_name)

        info_modules = self.info_modules
        for package_name,info_file in info_files:
            if info_modules.has_key(package_name):
                continue
            fullname = self.parent_name +'.'+ package_name
            if info_file[-1]=='c':
                filedescriptor = ('.pyc','rb',2)
            else:
                filedescriptor = ('.py','U',1)

            try:
                info_module = imp.load_module(fullname+'.info',
                                              open(info_file,filedescriptor[1]),
                                              info_file,
                                              filedescriptor)
            except Exception,msg:
                self.error(msg)
                info_module = None

            if info_module is None or getattr(info_module,'ignore',False):
                info_modules.pop(package_name,None)
            else:
                self._init_info_modules(getattr(info_module,'depends',[]))
                info_modules[package_name] = info_module

        return

    def _get_sorted_names(self):
        """ Return package names sorted in the order as they should be
        imported due to dependence relations between packages. 
        """

        depend_dict = {}
        for name,info_module in self.info_modules.items():
            depend_dict[name] = getattr(info_module,'depends',[])
        package_names = []

        for name in depend_dict.keys():
            if not depend_dict[name]:
                package_names.append(name)
                del depend_dict[name]

        while depend_dict:
            for name, lst in depend_dict.items():
                new_lst = [n for n in lst if depend_dict.has_key(n)]
                if not new_lst:
                    package_names.append(name)
                    del depend_dict[name]
                else:
                    depend_dict[name] = new_lst

        return package_names

    def __call__(self,*packages, **options):
        """Load one or more packages into numpy's top-level namespace.

    Usage:

       This function is intended to shorten the need to import many of numpy's
       submodules constantly with statements such as

