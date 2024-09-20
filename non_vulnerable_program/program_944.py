import scipy.f2py as f2py
f2py.main()
'''%(os.path.basename(sys.executable)))
            f.close()
        return target

    config.add_scripts(generate_f2py_py)

    print 'F2PY Version',config.get_version()

    return config

if __name__ == "__main__":

    config = configuration(top_path='')
    version = config.get_version()
    print 'F2PY Version',version
    config = config.todict()

    if sys.version[:3]>='2.3':
        config['download_url'] = "http://cens.ioc.ee/projects/f2py2e/2.x"\
                                 "/F2PY-2-latest.tar.gz"
        config['classifiers'] = [
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
            'Natural Language :: English',
            'Operating System :: OS Independent',
            'Programming Language :: C',
            'Programming Language :: Fortran',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering',
            'Topic :: Software Development :: Code Generators',
            ]
    setup(version=version,
          description       = "F2PY - Fortran to Python Interface Generaton",
          author            = "Pearu Peterson",
          author_email      = "pearu@cens.ioc.ee",
          maintainer        = "Pearu Peterson",
          maintainer_email  = "pearu@cens.ioc.ee",
          license           = "LGPL",
          platforms         = "Unix, Windows (mingw|cygwin), Mac OSX",
          long_description  = """\
The Fortran to Python Interface Generator, or F2PY for short, is a
command line tool (f2py) for generating Python C/API modules for
wrapping Fortran 77/90/95 subroutines, accessing common blocks from
Python, and calling Python functions from Fortran (call-backs).
Interfacing subroutines/data from Fortran 90/95 modules is supported.""",
          url               = "http://cens.ioc.ee/projects/f2py2e/",
          keywords          = ['Fortran','f2py'],
          **config)


""" Tools for compiling C/C++ code to extension modules

    The main function, build_extension(), takes the C/C++ file
    along with some other options and builds a Python extension.
    It uses distutils for most of the heavy lifting.
    
    choose_compiler() is also useful (mainly on windows anyway)
    for trying to determine whether MSVC++ or gcc is available.
    MSVC doesn't handle templates as well, so some of the code emitted
    by the python->C conversions need this info to choose what kind
    of code to create.
    
    The other main thing here is an alternative version of the MingW32
    compiler class.  The class makes it possible to build libraries with
    gcc even if the original version of python was built using MSVC.  It
    does this by converting a pythonxx.lib file to a libpythonxx.a file.
    Note that you need write access to the pythonxx/lib directory to do this.
"""

