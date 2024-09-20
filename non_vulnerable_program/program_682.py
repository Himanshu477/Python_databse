import os
from distutils.command.install import *
from distutils.command.install_headers import install_headers as old_install_headers

class install_headers (old_install_headers):

    def run (self):
        headers = self.distribution.headers
        if not headers:
            return

        prefix = os.path.dirname(self.install_dir)
        for header in headers:
            if isinstance(header,tuple):
                # Kind of a hack, but I don't know where else to change this...
                if header[0] == 'scipy.base':
                    header = ('scipy', header[1])
                    if os.path.splitext(header[1])[1] == '.inc':
                        continue
                d = os.path.join(*([prefix]+header[0].split('.')))
                header = header[1]
            else:
                d = self.install_dir
            self.mkpath(d)
            (out, _) = self.copy_file(header, d)
            self.outfiles.append(out)


"""
Unit-testing
------------

  ScipyTest -- Scipy tests site manager
  ScipyTestCase -- unittest.TestCase with measure method
  IgnoreException -- raise when checking disabled feature ('ignoring' is displayed)
  set_package_path -- prepend package build directory to path
  set_local_path -- prepend local directory (to tests files) to path
  restore_path -- restore path after set_package_path

Timing tools
------------

  jiffies -- return 1/100ths of a second that the current process has used
  memusage -- virtual memory size in bytes of the running python [linux]

Utility functions
-----------------

  assert_equal -- assert equality
  assert_almost_equal -- assert equality with decimal tolerance
  assert_approx_equal -- assert equality with significant digits tolerance
  assert_array_equal -- assert arrays equality
  assert_array_almost_equal -- assert arrays equality with decimal tolerance
  assert_array_less -- assert arrays less-ordering
  rand -- array of random numbers from given shape

"""

__all__ = []

