from utils import *
from arraysetops import *
from io import *
import math

__all__ = ['emath','math']
__all__ += type_check.__all__
__all__ += index_tricks.__all__
__all__ += function_base.__all__
__all__ += shape_base.__all__
__all__ += twodim_base.__all__
__all__ += ufunclike.__all__
__all__ += polynomial.__all__
__all__ += machar.__all__
__all__ += getlimits.__all__
__all__ += utils.__all__
__all__ += arraysetops.__all__
__all__ += io.__all__

def test(level=1, verbosity=1):
    from numpy.testing import NumpyTest
    return NumpyTest().test(level, verbosity)


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

