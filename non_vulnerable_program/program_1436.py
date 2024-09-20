import os
from distutils.command.install import *
from distutils.command.install import install as old_install
from distutils.file_util import write_file

class install(old_install):

    def finalize_options (self):
        old_install.finalize_options(self)
        self.install_lib = self.install_libbase

    def run(self):
        r = old_install.run(self)
        if self.record:
            # bdist_rpm fails when INSTALLED_FILES contains
            # paths with spaces. Such paths must be enclosed
            # with double-quotes.
            f = open(self.record,'r')
            lines = []
            need_rewrite = False
            for l in f.readlines():
                l = l.rstrip()
                if ' ' in l:
                    need_rewrite = True
                    l = '"%s"' % (l)
                lines.append(l)
            f.close()
            if need_rewrite:
                self.execute(write_file,
                             (self.record, lines),
                             "re-writing list of installed files to '%s'" %
                             self.record)
        return r


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

