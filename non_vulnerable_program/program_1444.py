from distutils.command.install_data import install_data as old_install_data

#data installer with improved intelligence over distutils
#data files are copied into the project directory instead
#of willy-nilly
class install_data (old_install_data):

    def finalize_options (self):
        self.set_undefined_options('install',
                                   ('install_lib', 'install_dir'),
                                   ('root', 'root'),
                                   ('force', 'force'),
                                  )




#
#  Calculate prime numbers
#

def primes(int kmax):
  cdef int n, k, i
  cdef int p[1000]
  result = []
  if kmax > 1000:
    kmax = 1000
  k = 0
  n = 2
  while k < kmax:
    i = 0
    while i < k and n % p[i] <> 0:
      i = i + 1
    if i == k:
      p[k] = n
      k = k + 1
      result.append(n)
    n = n + 1
  return result


#!/usr/bin/env python
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('pyrex_ext',parent_package,top_path)
    config.add_extension('primes',
                         ['primes.pyx'])
    return config

if __name__ == "__main__":
