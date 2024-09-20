    from numpy.f2py import main
else:
    print >> sys.stderr, "Unknown mode:",`mode`
    sys.exit(1)
main()
'''%(os.path.basename(sys.executable)))
            f.close()
        return target

    config.add_scripts(generate_f2py_py)

    log.info('F2PY Version %s', config.get_version())

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
            'License :: OSI Approved :: NumPy License',
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
          license           = "BSD",
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



def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    return Configuration('oldnumeric',parent_package,top_path)

if __name__ == '__main__':
