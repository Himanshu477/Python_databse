        import new
        f.seek = new.instancemethod(seek, f)
        f.tell = new.instancemethod(tell, f)

    return f

class BagObj(object):
    """
    BagObj(obj)

    Convert attribute lookups to getitems on the object passed in.

    Parameters
    ----------
    obj : class instance
        Object on which attribute lookup is performed.

    Examples
    --------
    >>> class BagDemo(object):
    ...     def __getitem__(self, key):
    ...         return key
    ...
    >>> demo_obj = BagDemo()
    >>> bagobj = np.lib.npyio.BagObj(demo_obj)
    >>> bagobj.some_item
    'some_item'

    """
    def __init__(self, obj):
        self._obj = obj
    def __getattribute__(self, key):
        try:
            return object.__getattribute__(self, '_obj')[key]
        except KeyError:
            raise AttributeError, key

class NpzFile(object):
    """
    NpzFile(fid)

    A dictionary-like object with lazy-loading of files in the zipped
    archive provided on construction.

    `NpzFile` is used to load files in the NumPy ``.npz`` data archive
    format. It assumes that files in the archive have a ".npy" extension,
    other files are ignored.

    The arrays and file strings are lazily loaded on either
    getitem access using ``obj['key']`` or attribute lookup using
    ``obj.f.key``. A list of all files (without ".npy" extensions) can
    be obtained with ``obj.files`` and the ZipFile object itself using
    ``obj.zip``.

    Attributes
    ----------
    files : list of str
        List of all files in the archive with a ".npy" extension.
    zip : ZipFile instance
        The ZipFile object initialized with the zipped archive.
    f : BagObj instance
        An object on which attribute can be performed as an alternative
        to getitem access on the `NpzFile` instance itself.

    Parameters
    ----------
    fid : file or str
        The zipped archive to open. This is either a file-like object
        or a string containing the path to the archive.

    Examples
    --------
    >>> from tempfile import TemporaryFile
    >>> outfile = TemporaryFile()
    >>> x = np.arange(10)
    >>> y = np.sin(x)
    >>> np.savez(outfile, x=x, y=y)
    >>> outfile.seek(0)

    >>> npz = np.load(outfile)
    >>> isinstance(npz, np.lib.npyio.NpzFile)
    True
    >>> npz.files
    ['y', 'x']
    >>> npz['x']  # getitem access
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> npz.f.x  # attribute lookup
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    """
    def __init__(self, fid):
        # Import is postponed to here since zipfile depends on gzip, an optional
        # component of the so-called standard library.
        import zipfile
        _zip = zipfile.ZipFile(fid)
        self._files = _zip.namelist()
        self.files = []
        for x in self._files:
            if x.endswith('.npy'):
                self.files.append(x[:-4])
            else:
                self.files.append(x)
        self.zip = _zip
        self.f = BagObj(self)

    def __getitem__(self, key):
        # FIXME: This seems like it will copy strings around
        #   more than is strictly necessary.  The zipfile
        #   will read the string and then
        #   the format.read_array will copy the string
        #   to another place in memory.
        #   It would be better if the zipfile could read
        #   (or at least uncompress) the data
        #   directly into the array memory.
        member = 0
        if key in self._files:
            member = 1
        elif key in self.files:
            member = 1
            key += '.npy'
        if member:
            bytes = self.zip.read(key)
            if bytes.startswith(format.MAGIC_PREFIX):
                value = BytesIO(bytes)
                return format.read_array(value)
            else:
                return bytes
        else:
            raise KeyError, "%s is not a file in the archive" % key


    def __iter__(self):
        return iter(self.files)

    def items(self):
        """
        Return a list of tuples, with each tuple (filename, array in file).

        """
        return [(f, self[f]) for f in self.files]

    def iteritems(self):
        """Generator that returns tuples (filename, array in file)."""
        for f in self.files:
            yield (f, self[f])

    def keys(self):
        """Return files in the archive with a ".npy" extension."""
        return self.files

    def iterkeys(self):
        """Return an iterator over the files in the archive."""
        return self.__iter__()

    def __contains__(self, key):
        return self.files.__contains__(key)


def load(file, mmap_mode=None):
    """
    Load a pickled, ``.npy``, or ``.npz`` binary file.

    Parameters
    ----------
    file : file-like object or string
        The file to read.  It must support ``seek()`` and ``read()`` methods.
        If the filename extension is ``.gz``, the file is first decompressed.
    mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
        If not None, then memory-map the file, using the given mode
        (see `numpy.memmap`).  The mode has no effect for pickled or
        zipped files.
        A memory-mapped array is stored on disk, and not directly loaded
        into memory.  However, it can be accessed and sliced like any
        ndarray.  Memory mapping is especially useful for accessing
        small fragments of large files without reading the entire file
        into memory.

    Returns
    -------
    result : array, tuple, dict, etc.
        Data stored in the file.

    Raises
    ------
    IOError
        If the input file does not exist or cannot be read.

    See Also
    --------
    save, savez, loadtxt
    memmap : Create a memory-map to an array stored in a file on disk.

    Notes
    -----
    - If the file contains pickle data, then whatever is stored in the
      pickle is returned.
    - If the file is a ``.npy`` file, then an array is returned.
    - If the file is a ``.npz`` file, then a dictionary-like object is
      returned, containing ``{filename: array}`` key-value pairs, one for
      each file in the archive.

    Examples
    --------
    Store data to disk, and load it again:

    >>> np.save('/tmp/123', np.array([[1, 2, 3], [4, 5, 6]]))
    >>> np.load('/tmp/123.npy')
    array([[1, 2, 3],
           [4, 5, 6]])

    Mem-map the stored array, and then access the second row
    directly from disk:

    >>> X = np.load('/tmp/123.npy', mmap_mode='r')
    >>> X[1, :]
    memmap([4, 5, 6])

    """
    import gzip

    if isinstance(file, basestring):
        fid = _file(file, "rb")
    elif isinstance(file, gzip.GzipFile):
        fid = seek_gzip_factory(file)
    else:
        fid = file

    # Code to distinguish from NumPy binary files and pickles.
    _ZIP_PREFIX = asbytes('PK\x03\x04')
    N = len(format.MAGIC_PREFIX)
    magic = fid.read(N)
    fid.seek(-N, 1) # back-up
    if magic.startswith(_ZIP_PREFIX):  # zip-file (assume .npz)
        return NpzFile(fid)
    elif magic == format.MAGIC_PREFIX: # .npy file
        if mmap_mode:
            return format.open_memmap(file, mode=mmap_mode)
        else:
            return format.read_array(fid)
    else:  # Try a pickle
        try:
            return _cload(fid)
        except:
            raise IOError, \
                "Failed to interpret file %s as a pickle" % repr(file)

def save(file, arr):
    """
    Save an array to a binary file in NumPy ``.npy`` format.

    Parameters
    ----------
    file : file or str
        File or filename to which the data is saved.  If file is a file-object,
        then the filename is unchanged.  If file is a string, a ``.npy``
        extension will be appended to the file name if it does not already
        have one.
    arr : array_like
        Array data to be saved.

    See Also
    --------
    savez : Save several arrays into a ``.npz`` compressed archive
    savetxt, load

    Notes
    -----
    For a description of the ``.npy`` format, see `format`.

    Examples
    --------
    >>> from tempfile import TemporaryFile
    >>> outfile = TemporaryFile()

    >>> x = np.arange(10)
    >>> np.save(outfile, x)

    >>> outfile.seek(0) # Only needed here to simulate closing & reopening file
    >>> np.load(outfile)
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    """
    if isinstance(file, basestring):
        if not file.endswith('.npy'):
            file = file + '.npy'
        fid = open(file, "wb")
    else:
        fid = file

    arr = np.asanyarray(arr)
    format.write_array(fid, arr)

def savez(file, *args, **kwds):
    """
    Save several arrays into a single, compressed file in ``.npz`` format.

    If arguments are passed in with no keywords, the corresponding variable
    names, in the .npz file, are 'arr_0', 'arr_1', etc. If keyword arguments
    are given, the corresponding variable names, in the ``.npz`` file will
    match the keyword names.

    Parameters
    ----------
    file : str or file
        Either the file name (string) or an open file (file-like object)
        where the data will be saved. If file is a string, the ``.npz``
        extension will be appended to the file name if it is not already there.
    \\*args : Arguments, optional
        Arrays to save to the file. Since it is not possible for Python to
        know the names of the arrays outside `savez`, the arrays will be saved
        with names "arr_0", "arr_1", and so on. These arguments can be any
        expression.
    \\*\\*kwds : Keyword arguments, optional
        Arrays to save to the file. Arrays will be saved in the file with the
        keyword names.

    Returns
    -------
    None

    See Also
    --------
    save : Save a single array to a binary file in NumPy format.
    savetxt : Save an array to a file as plain text.

    Notes
    -----
    The ``.npz`` file format is a zipped archive of files named after the
    variables they contain.  Each file contains one variable in ``.npy``
    format. For a description of the ``.npy`` format, see `format`.

    When opening the saved ``.npz`` file with `load` a `NpzFile` object is
    returned. This is a dictionary-like object which can be queried for
    its list of arrays (with the ``.files`` attribute), and for the arrays
    themselves.

    Examples
    --------
    >>> from tempfile import TemporaryFile
    >>> outfile = TemporaryFile()
    >>> x = np.arange(10)
    >>> y = np.sin(x)

    Using `savez` with \\*args, the arrays are saved with default names.

    >>> np.savez(outfile, x, y)
    >>> outfile.seek(0) # Only needed here to simulate closing & reopening file
    >>> npzfile = np.load(outfile)
    >>> npzfile.files
    ['arr_1', 'arr_0']
    >>> npzfile['arr_0']
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    Using `savez` with \\*\\*kwds, the arrays are saved with the keyword names.

    >>> outfile = TemporaryFile()
    >>> np.savez(outfile, x=x, y=y)
    >>> outfile.seek(0)
    >>> npzfile = np.load(outfile)
    >>> npzfile.files
    ['y', 'x']
    >>> npzfile['x']
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    """

    # Import is postponed to here since zipfile depends on gzip, an optional
    # component of the so-called standard library.
    import zipfile
    # Import deferred for startup time improvement
    import tempfile

    if isinstance(file, basestring):
        if not file.endswith('.npz'):
            file = file + '.npz'

    namedict = kwds
    for i, val in enumerate(args):
        key = 'arr_%d' % i
        if key in namedict.keys():
            raise ValueError, "Cannot use un-named variables and keyword %s" % key
        namedict[key] = val

    zip = zipfile.ZipFile(file, mode="w")

    # Stage arrays in a temporary file on disk, before writing to zip.
    fd, tmpfile = tempfile.mkstemp(suffix='-numpy.npy')
    os.close(fd)
    try:
        for key, val in namedict.iteritems():
            fname = key + '.npy'
            fid = open(tmpfile, 'wb')
            try:
                format.write_array(fid, np.asanyarray(val))
                fid.close()
                fid = None
                zip.write(tmpfile, arcname=fname)
            finally:
                if fid:
                    fid.close()
    finally:
        os.remove(tmpfile)

    zip.close()

# Adapted from matplotlib

def _getconv(dtype):
    typ = dtype.type
    if issubclass(typ, np.bool_):
        return lambda x: bool(int(x))
    if issubclass(typ, np.integer):
        return lambda x: int(float(x))
    elif issubclass(typ, np.floating):
        return float
    elif issubclass(typ, np.complex):
        return complex
    elif issubclass(typ, np.bytes_):
        return bytes
    else:
        return str



def loadtxt(fname, dtype=float, comments='#', delimiter=None,
            converters=None, skiprows=0, usecols=None, unpack=False):
    """
    Load data from a text file.

    Each row in the text file must have the same number of values.

    Parameters
    ----------
    fname : file or str
        File or filename to read.  If the filename extension is ``.gz`` or
        ``.bz2``, the file is first decompressed.
    dtype : dtype, optional
        Data type of the resulting array.  If this is a record data-type,
        the resulting array will be 1-dimensional, and each row will be
        interpreted as an element of the array.   In this case, the number
        of columns used must match the number of fields in the data-type.
    comments : str, optional
        The character used to indicate the start of a comment.
    delimiter : str, optional
        The string used to separate values.  By default, this is any
        whitespace.
    converters : dict, optional
        A dictionary mapping column number to a function that will convert
        that column to a float.  E.g., if column 0 is a date string:
        ``converters = {0: datestr2num}``. Converters can also be used to
        provide a default value for missing data:
        ``converters = {3: lambda s: float(s or 0)}``.
    skiprows : int, optional
        Skip the first `skiprows` lines.
    usecols : sequence, optional
        Which columns to read, with 0 being the first.  For example,
        ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns.
    unpack : bool, optional
        If True, the returned array is transposed, so that arguments may be
        unpacked using ``x, y, z = loadtxt(...)``. Default is False.

    Returns
    -------
    out : ndarray
        Data read from the text file.

    See Also
    --------
    load, fromstring, fromregex
    genfromtxt : Load data with missing values handled as specified.
    scipy.io.loadmat : reads Matlab(R) data files

    Notes
    -----
    This function aims to be a fast reader for simply formatted files. The
    `genfromtxt` function provides more sophisticated handling of, e.g.,
    lines with missing values.

    Examples
    --------
    >>> from StringIO import StringIO   # StringIO behaves like a file object
    >>> c = StringIO("0 1\\n2 3")
    >>> np.loadtxt(c)
    array([[ 0.,  1.],
           [ 2.,  3.]])

    >>> d = StringIO("M 21 72\\nF 35 58")
    >>> np.loadtxt(d, dtype={'names': ('gender', 'age', 'weight'),
    ...                      'formats': ('S1', 'i4', 'f4')})
    array([('M', 21, 72.0), ('F', 35, 58.0)],
          dtype=[('gender', '|S1'), ('age', '<i4'), ('weight', '<f4')])

    >>> c = StringIO("1,0,2\\n3,0,4")
    >>> x, y = np.loadtxt(c, delimiter=',', usecols=(0, 2), unpack=True)
    >>> x
    array([ 1.,  3.])
    >>> y
    array([ 2.,  4.])

    """
    # Type conversions for Py3 convenience
    comments = asbytes(comments)
    if delimiter is not None:
        delimiter = asbytes(delimiter)

    user_converters = converters

    if usecols is not None:
        usecols = list(usecols)

    isstring = False
    if _is_string_like(fname):
        isstring = True
        if fname.endswith('.gz'):
            import gzip
            fh = seek_gzip_factory(fname)
        elif fname.endswith('.bz2'):
            import bz2
            fh = bz2.BZ2File(fname)
        else:
            fh = file(fname)
    elif hasattr(fname, 'readline'):
        fh = fname
    else:
        raise ValueError('fname must be a string or file handle')
    X = []

    def flatten_dtype(dt):
        """Unpack a structured data-type."""
        if dt.names is None:
            # If the dtype is flattened, return.
            # If the dtype has a shape, the dtype occurs
            # in the list more than once.
            return [dt.base] * int(np.prod(dt.shape))
        else:
            types = []
            for field in dt.names:
                tp, bytes = dt.fields[field]
                flat_dt = flatten_dtype(tp)
                types.extend(flat_dt)
            return types

    def split_line(line):
        """Chop off comments, strip, and split at delimiter."""
        line = line.split(comments)[0].strip()
        if line:
            return line.split(delimiter)
        else:
            return []

    try:
        # Make sure we're dealing with a proper dtype
        dtype = np.dtype(dtype)
        defconv = _getconv(dtype)

        # Skip the first `skiprows` lines
        for i in xrange(skiprows):
            fh.readline()

        # Read until we find a line with some values, and use
        # it to estimate the number of columns, N.
        first_vals = None
        while not first_vals:
            first_line = fh.readline()
            if not first_line: # EOF reached
                raise IOError('End-of-file reached before encountering data.')
            first_vals = split_line(first_line)
        N = len(usecols or first_vals)

        dtype_types = flatten_dtype(dtype)
        if len(dtype_types) > 1:
            # We're dealing with a structured array, each field of
            # the dtype matches a column
            converters = [_getconv(dt) for dt in dtype_types]
        else:
            # All fields have the same dtype
            converters = [defconv for i in xrange(N)]

        # By preference, use the converters specified by the user
        for i, conv in (user_converters or {}).iteritems():
            if usecols:
                try:
                    i = usecols.index(i)
                except ValueError:
                    # Unused converter specified
                    continue
            converters[i] = conv

        # Parse each line, including the first
        for i, line in enumerate(itertools.chain([first_line], fh)):
            vals = split_line(line)
            if len(vals) == 0:
                continue

            if usecols:
                vals = [vals[i] for i in usecols]

            # Convert each value according to its column and store
            X.append(tuple([conv(val) for (conv, val) in zip(converters, vals)]))
    finally:
        if isstring:
            fh.close()

    if len(dtype_types) > 1:
        # We're dealing with a structured array, with a dtype such as
        # [('x', int), ('y', [('s', int), ('t', float)])]
        #
        # First, create the array using a flattened dtype:
        # [('x', int), ('s', int), ('t', float)]
        #
        # Then, view the array using the specified dtype.
        try:
            X = np.array(X, dtype=np.dtype([('', t) for t in dtype_types]))
            X = X.view(dtype)
        except TypeError:
            # In the case we have an object dtype
            X = np.array(X, dtype=dtype)
    else:
        X = np.array(X, dtype)

    X = np.squeeze(X)
    if unpack:
        return X.T
    else:
        return X


def savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n'):
    """
    Save an array to a text file.

    Parameters
    ----------
    fname : filename or file handle
        If the filename ends in ``.gz``, the file is automatically saved in
        compressed gzip format.  `loadtxt` understands gzipped files
        transparently.
    X : array_like
        Data to be saved to a text file.
    fmt : str or sequence of strs
        A single format (%10.5f), a sequence of formats, or a
        multi-format string, e.g. 'Iteration %d -- %10.5f', in which
        case `delimiter` is ignored.
    delimiter : str
        Character separating columns.
    newline : str
        .. versionadded:: 2.0

        Character separating lines.


    See Also
    --------
    save : Save an array to a binary file in NumPy ``.npy`` format
    savez : Save several arrays into a ``.npz`` compressed archive

    Notes
    -----
    Further explanation of the `fmt` parameter
    (``%[flag]width[.precision]specifier``):

    flags:
        ``-`` : left justify

        ``+`` : Forces to preceed result with + or -.

        ``0`` : Left pad the number with zeros instead of space (see width).

    width:
        Minimum number of characters to be printed. The value is not truncated
        if it has more characters.

    precision:
        - For integer specifiers (eg. ``d,i,o,x``), the minimum number of
          digits.
        - For ``e, E`` and ``f`` specifiers, the number of digits to print
          after the decimal point.
        - For ``g`` and ``G``, the maximum number of significant digits.
        - For ``s``, the maximum number of characters.

    specifiers:
        ``c`` : character

        ``d`` or ``i`` : signed decimal integer

        ``e`` or ``E`` : scientific notation with ``e`` or ``E``.

        ``f`` : decimal floating point

        ``g,G`` : use the shorter of ``e,E`` or ``f``

        ``o`` : signed octal

        ``s`` : string of characters

        ``u`` : unsigned decimal integer

        ``x,X`` : unsigned hexadecimal integer

    This explanation of ``fmt`` is not complete, for an exhaustive
    specification see [1]_.

    References
    ----------
    .. [1] `Format Specification Mini-Language
           <http://docs.python.org/library/string.html#
           format-specification-mini-language>`_, Python Documentation.

    Examples
    --------
    >>> x = y = z = np.arange(0.0,5.0,1.0)
    >>> np.savetxt('test.out', x, delimiter=',')   # X is an array
    >>> np.savetxt('test.out', (x,y,z))   # x,y,z equal sized 1D arrays
    >>> np.savetxt('test.out', x, fmt='%1.4e')   # use exponential notation

    """

    # Py3 conversions first
    if isinstance(fmt, bytes):
        fmt = asstr(fmt)
    delimiter = asstr(delimiter)

    if _is_string_like(fname):
        if fname.endswith('.gz'):
            import gzip
            fh = gzip.open(fname, 'wb')
        else:
            if sys.version_info[0] >= 3:
                fh = file(fname, 'wb')
            else:
                fh = file(fname, 'w')
    elif hasattr(fname, 'seek'):
        fh = fname
    else:
        raise ValueError('fname must be a string or file handle')

    X = np.asarray(X)

    # Handle 1-dimensional arrays
    if X.ndim == 1:
        # Common case -- 1d array of numbers
        if X.dtype.names is None:
            X = np.atleast_2d(X).T
            ncol = 1

        # Complex dtype -- each field indicates a separate column
        else:
            ncol = len(X.dtype.descr)
    else:
        ncol = X.shape[1]

    # `fmt` can be a string with multiple insertion points or a list of formats.
    # E.g. '%10.5f\t%10d' or ('%10.5f', '$10d')
    if type(fmt) in (list, tuple):
        if len(fmt) != ncol:
            raise AttributeError('fmt has wrong shape.  %s' % str(fmt))
        format = asstr(delimiter).join(map(asstr, fmt))
    elif type(fmt) is str:
        if fmt.count('%') == 1:
            fmt = [fmt, ]*ncol
            format = delimiter.join(fmt)
        elif fmt.count('%') != ncol:
            raise AttributeError('fmt has wrong number of %% formats.  %s'
                                 % fmt)
        else:
            format = fmt

    for row in X:
        fh.write(asbytes(format % tuple(row) + newline))

