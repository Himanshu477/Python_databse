    import os,sys
    from distutils.core import setup
    print 'SciPy core Version %s' % scipy_core_version
    setup (name = "SciPy_core",
           version = scipy_core_version,
           maintainer = "SciPy Developers",
           maintainer_email = "scipy-dev@scipy.org",
           description = "SciPy core modules: scipy_test and scipy_distutils",
           license = "SciPy License (BSD Style)",
           url = "http://www.scipy.org",
           packages=['scipy_distutils',
                     'scipy_distutils.command',
                     'scipy_test'],
           package_dir = {'scipy_distutils':'scipy_distutils',
                          'scipy_test':'scipy_test',
                          },
           )


