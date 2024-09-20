from __version__ import version

def configuration(parent_package='',top_path=None):
    config = Configuration('f2py', parent_package, top_path)

    config.add_subpackage('lib')

    config.add_data_dir('docs')

    config.add_data_files('src/fortranobject.c',
                          'src/fortranobject.h',
                          'f2py.1'
                          )

    config.make_svn_version_py()

    def generate_f2py_py(build_dir):
        f2py_exe = 'f2py'+os.path.basename(sys.executable)[6:]
        if f2py_exe[-4:]=='.exe':
            f2py_exe = f2py_exe[:-4] + '.py'
        if 'bdist_wininst' in sys.argv and f2py_exe[-3:] != '.py':
            f2py_exe = f2py_exe + '.py'
        target = os.path.join(build_dir,f2py_exe)
        if newer(__file__,target):
            log.info('Creating %s', target)
            f = open(target,'w')
            f.write('''\
#!/usr/bin/env %s
# See http://cens.ioc.ee/projects/f2py2e/
