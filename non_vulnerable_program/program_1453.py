    import numpy

    temp=reshape(arange(10000),(100,100))

    ua=UserArray(temp)
    # new object created begin test
    print dir(ua)
    print shape(ua),ua.shape # I have changed Numeric.py

    ua_small=ua[:3,:5]
    print ua_small
    ua_small[0,0]=10  # this did not change ua[0,0], which is not normal behavior
    print ua_small[0,0],ua[0,0]
    print sin(ua_small)/3.*6.+sqrt(ua_small**2)
    print less(ua_small,103),type(less(ua_small,103))
    print type(ua_small*reshape(arange(15),shape(ua_small)))
    print reshape(ua_small,(5,3))
    print transpose(ua_small)


"""
Scipy testing tools
===================

Scipy-style unit-testing
------------------------

  ScipyTest -- Scipy tests site manager
  ScipyTestCase -- unittest.TestCase with measure method
  IgnoreException -- raise when checking disabled feature, it'll be ignored
  set_package_path -- prepend package build directory to path
  set_local_path -- prepend local directory (to tests files) to path
  restore_path -- restore path after set_package_path

Utility functions
-----------------

  jiffies -- return 1/100ths of a second that the current process has used
  memusage -- virtual memory size in bytes of the running python [linux]
  rand -- array of random numbers from given shape
  assert_equal -- assert equality
  assert_almost_equal -- assert equality with decimal tolerance
  assert_approx_equal -- assert equality with significant digits tolerance
  assert_array_equal -- assert arrays equality
  assert_array_almost_equal -- assert arrays equality with decimal tolerance
  assert_array_less -- assert arrays less-ordering

"""

global_symbols = ['ScipyTest','NumpyTest']


"""distutils.command

Package containing implementation of all the standard Distutils
commands."""

__revision__ = "$Id: __init__.py,v 1.3 2005/05/16 11:08:49 pearu Exp $"

distutils_all = [  'build_py',
                   'clean',
                   'install_lib',
                   'install_scripts',
                   'bdist',
                   'bdist_dumb',
                   'bdist_wininst',
                ]

__import__('distutils.command',globals(),locals(),distutils_all)

__all__ = ['build',
           'config_compiler',
           'config',
           'build_src',
           'build_ext',
           'build_clib',
           'build_scripts',
           'install',
           'install_data',
           'install_headers',
           'bdist_rpm',
           'sdist',
          ] + distutils_all


#!/usr/bin/python

# takes templated file .xxx.src and produces .xxx file  where .xxx is .i or .c or .h
#  using the following template rules

# /**begin repeat     on a line by itself marks the beginning of a segment of code to be repeated
# /**end repeat**/    on a line by itself marks it's end

# after the /**begin repeat and before the */
#  all the named templates are placed
#  these should all have the same number of replacements

#  in the main body, the names are used.
#  Each replace will use one entry from the list of named replacements

#  Note that all #..# forms in a block must have the same number of
#    comma-separated entries.

__all__ = ['process_str', 'process_file']

