import os
from distutils.dist import Distribution

__metaclass__ = type

class EnvironmentConfig:
    def __init__(self, distutils_section='DEFAULT', **kw):
        self._distutils_section = distutils_section
        self._conf_keys = kw
        self._conf = None
        self._hook_handler = None

    def __getattr__(self, name):
        try:
            conf_desc = self._conf_keys[name]
        except KeyError:
            raise AttributeError(name)
        return self._get_var(name, conf_desc)

    def get(self, name, default=None):
        try:
            conf_desc = self._conf_keys[name]
        except KeyError:
            return default
        var = self._get_var(name, conf_desc)
        if var is None:
            var = default
        return var

    def _get_var(self, name, conf_desc):
        hook, envvar, confvar = conf_desc
        var = self._hook_handler(name, hook)
        if envvar is not None:
            var = os.environ.get(envvar, var)
        if confvar is not None and self._conf:
            var = self._conf.get(confvar, (None, var))[1]
        return var

    def clone(self, hook_handler):
        ec = self.__class__(distutils_section=self._distutils_section,
                            **self._conf_keys)
        ec._hook_handler = hook_handler
        return ec

    def use_distribution(self, dist):
        if isinstance(dist, Distribution):
            self._conf = dist.get_option_dict(self._distutils_section)
        else:
            self._conf = dist


#!/usr/bin/env python
"""
exec_command

Implements exec_command function that is (almost) equivalent to
commands.getstatusoutput function but on NT, DOS systems the
returned status is actually correct (though, the returned status
values may be different by a factor). In addition, exec_command
takes keyword arguments for (re-)defining environment variables.

Provides functions:
  exec_command  --- execute command in a specified directory and
                    in the modified environment.
  splitcmdline  --- inverse of ' '.join(argv)
  find_executable --- locate a command using info from environment
                    variable PATH. Equivalent to posix `which`
                    command.

Author: Pearu Peterson <pearu@cens.ioc.ee>
Created: 11 January 2003

Requires: Python 2.x

Succesfully tested on:
  os.name | sys.platform | comments
  --------+--------------+----------
  posix   | linux2       | Debian (sid) Linux, Python 2.1.3+, 2.2.3+, 2.3.3
                           PyCrust 0.9.3, Idle 1.0.2
  posix   | linux2       | Red Hat 9 Linux, Python 2.1.3, 2.2.2, 2.3.2
  posix   | sunos5       | SunOS 5.9, Python 2.2, 2.3.2
  posix   | darwin       | Darwin 7.2.0, Python 2.3
  nt      | win32        | Windows Me
                           Python 2.3(EE), Idle 1.0, PyCrust 0.7.2
                           Python 2.1.1 Idle 0.8
  nt      | win32        | Windows 98, Python 2.1.1. Idle 0.8
  nt      | win32        | Cygwin 98-4.10, Python 2.1.1(MSC) - echo tests
                           fail i.e. redefining environment variables may
                           not work. FIXED: don't use cygwin echo!
                           Comment: also `cmd /c echo` will not work
                           but redefining environment variables do work.
  posix   | cygwin       | Cygwin 98-4.10, Python 2.3.3(cygming special)
  nt      | win32        | Windows XP, Python 2.3.3

Known bugs:
- Tests, that send messages to stderr, fail when executed from MSYS prompt
  because the messages are lost at some point.
"""

__all__ = ['exec_command','find_executable']

