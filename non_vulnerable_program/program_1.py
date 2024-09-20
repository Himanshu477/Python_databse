

"""scipy_distutils

   Modified version of distutils to handle fortran source code, f2py,
   and other issues in the scipy build process.
"""

# Need to do something here to get distutils subsumed...


"""distutils.command

Package containing implementation of all the standard Distutils
commands."""

__revision__ = "$Id$"

distutils_all = [  'build_py',
                   'build_scripts',
                   'clean',
                   'install_lib',
                   'install_scripts',
                   'bdist',
                   'bdist_dumb',
                   'bdist_rpm',
                   'bdist_wininst',
                ]

__import__('distutils.command',globals(),locals(),distutils_all)

__all__ = ['build',
           'build_ext',
           'build_clib',
           'build_flib',
           'install',
           'install_data',
           'install_headers',
           'sdist',
          ] + distutils_all


# Need to override the build command to include building of fortran libraries
# This class must be used as the entry for the build key in the cmdclass
#    dictionary which is given to the setup command.

