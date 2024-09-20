    from Cython.Distutils import build_ext
    has_cython = True
except ImportError:
    has_cython = False

# Define a cython-based extension module, using the generated sources if cython
# is not available.
if has_cython:
    pyx_sources = ['numpyx.pyx']
    cmdclass    = {'build_ext': build_ext}
else:
    # In production work, you can ship the auto-generated C source yourself to
    # your users.  In this case, we do NOT ship the .c file as part of numpy,
    # so you'll need to actually have cython installed at least the first
    # time.  Since this is really just an example to show you how to use
    # *Cython*, it makes more sense NOT to ship the C sources so you can edit
    # the pyx at will with less chances for source update conflicts when you
    # update numpy.
    pyx_sources = ['numpyx.c']
    cmdclass    = {}


# Declare the extension object
pyx_ext = Extension('numpyx',
                    pyx_sources,
                    include_dirs = [numpy.get_include()])

# Call the routine which does the real work
setup(name        = 'numpyx',
      description = 'Small example on using Cython to write a Numpy extension',
      ext_modules = [pyx_ext],
      cmdclass    = cmdclass,
      )


"""A file interface for handling local and remote data files.
The goal of datasource is to abstract some of the file system operations when
dealing with data files so the researcher doesn't have to know all the
low-level details.  Through datasource, a researcher can obtain and use a
file with one function call, regardless of location of the file.

DataSource is meant to augment standard python libraries, not replace them.
It should work seemlessly with standard file IO operations and the os module.

DataSource files can originate locally or remotely:

- local files : '/home/guido/src/local/data.txt'
- URLs (http, ftp, ...) : 'http://www.scipy.org/not/real/data.txt'

DataSource files can also be compressed or uncompressed.  Currently only gzip
and bz2 are supported.

Example:

    >>> # Create a DataSource, use os.curdir (default) for local storage.
    >>> ds = datasource.DataSource()
    >>>
    >>> # Open a remote file.
    >>> # DataSource downloads the file, stores it locally in:
    >>> #     './www.google.com/index.html'
    >>> # opens the file and returns a file object.
    >>> fp = ds.open('http://www.google.com/index.html')
    >>>
    >>> # Use the file as you normally would
    >>> fp.read()
    >>> fp.close()

"""

__docformat__ = "restructuredtext en"

