import re
def fromregex(file, regexp, dtype):
    """
    Construct an array from a text file, using regular expression parsing.

    The returned array is always a structured array, and is constructed from
    all matches of the regular expression in the file. Groups in the regular
    expression are converted to fields of the structured array.

    Parameters
    ----------
    file : str or file
        File name or file object to read.
    regexp : str or regexp
        Regular expression used to parse the file.
        Groups in the regular expression correspond to fields in the dtype.
    dtype : dtype or list of dtypes
        Dtype for the structured array.

    Returns
    -------
    output : ndarray
        The output array, containing the part of the content of `file` that
        was matched by `regexp`. `output` is always a structured array.

    Raises
    ------
    TypeError
        When `dtype` is not a valid dtype for a structured array.

    See Also
    --------
    fromstring, loadtxt

    Notes
    -----
    Dtypes for structured arrays can be specified in several forms, but all
    forms specify at least the data type and field name. For details see
    `doc.structured_arrays`.

    Examples
    --------
    >>> f = open('test.dat', 'w')
    >>> f.write("1312 foo\\n1534  bar\\n444   qux")
    >>> f.close()

    >>> regexp = r"(\\d+)\\s+(...)"  # match [digits, whitespace, anything]
    >>> output = np.fromregex('test.dat', regexp,
    ...                       [('num', np.int64), ('key', 'S3')])
    >>> output
    array([(1312L, 'foo'), (1534L, 'bar'), (444L, 'qux')],
          dtype=[('num', '<i8'), ('key', '|S3')])
    >>> output['num']
    array([1312, 1534,  444], dtype=int64)

    """
    if not hasattr(file, "read"):
        file = open(file, 'rb')
    if not hasattr(regexp, 'match'):
        regexp = re.compile(asbytes(regexp))
    if not isinstance(dtype, np.dtype):
        dtype = np.dtype(dtype)

    seq = regexp.findall(file.read())
    if seq and not isinstance(seq[0], tuple):
        # Only one group is in the regexp.
        # Create the new array as a single data-type and then
        #   re-interpret as a single-field structured array.
        newdtype = np.dtype(dtype[dtype.names[0]])
        output = np.array(seq, dtype=newdtype)
        output.dtype = dtype
    else:
        output = np.array(seq, dtype=dtype)

    return output




#####--------------------------------------------------------------------------
#---- --- ASCII functions ---
#####--------------------------------------------------------------------------



def genfromtxt(fname, dtype=float, comments='#', delimiter=None,
               skiprows=0, skip_header=0, skip_footer=0, converters=None,
               missing='', missing_values=None, filling_values=None,
               usecols=None, names=None, excludelist=None, deletechars=None,
               autostrip=False, case_sensitive=True, defaultfmt="f%i",
               unpack=None, usemask=False, loose=True, invalid_raise=True):
    """
    Load data from a text file, with missing values handled as specified.

    Each line past the first `skiprows` lines is split at the `delimiter`
    character, and characters following the `comments` character are discarded.

    Parameters
    ----------
    fname : file or str
        File or filename to read.  If the filename extension is `.gz` or
        `.bz2`, the file is first decompressed.
    dtype : dtype, optional
        Data type of the resulting array.
        If None, the dtypes will be determined by the contents of each
        column, individually.
    comments : str, optional
        The character used to indicate the start of a comment.
        All the characters occurring on a line after a comment are discarded
    delimiter : str, int, or sequence, optional
        The string used to separate values.  By default, any consecutive
        whitespaces act as delimiter.  An integer or sequence of integers
        can also be provided as width(s) of each field.
    skip_header : int, optional
        The numbers of lines to skip at the beginning of the file.
    skip_footer : int, optional
        The numbers of lines to skip at the end of the file
    converters : variable or None, optional
        The set of functions that convert the data of a column to a value.
        The converters can also be used to provide a default value
        for missing data: ``converters = {3: lambda s: float(s or 0)}``.
    missing_values : variable or None, optional
        The set of strings corresponding to missing data.
    filling_values : variable or None, optional
        The set of values to be used as default when the data are missing.
    usecols : sequence or None, optional
        Which columns to read, with 0 being the first.  For example,
        ``usecols = (1, 4, 5)`` will extract the 2nd, 5th and 6th columns.
    names : {None, True, str, sequence}, optional
        If `names` is True, the field names are read from the first valid line
        after the first `skiprows` lines.
        If `names` is a sequence or a single-string of comma-separated names,
        the names will be used to define the field names in a structured dtype.
        If `names` is None, the names of the dtype fields will be used, if any.
    excludelist : sequence, optional
        A list of names to exclude. This list is appended to the default list
        ['return','file','print']. Excluded names are appended an underscore:
        for example, `file` would become `file_`.
    deletechars : str, optional
        A string combining invalid characters that must be deleted from the
        names.
    defaultfmt : str, optional
        A format used to define default field names, such as "f%i" or "f_%02i".
    autostrip : bool, optional
        Whether to automatically strip white spaces from the variables.
    case_sensitive : {True, False, 'upper', 'lower'}, optional
        If True, field names are case sensitive.
        If False or 'upper', field names are converted to upper case.
        If 'lower', field names are converted to lower case.
    unpack : bool, optional
        If True, the returned array is transposed, so that arguments may be
        unpacked using ``x, y, z = loadtxt(...)``
    usemask : bool, optional
        If True, return a masked array.
        If False, return a regular array.
    invalid_raise : bool, optional
        If True, an exception is raised if an inconsistency is detected in the
        number of columns.
        If False, a warning is emitted and the offending lines are skipped.

    Returns
    -------
    out : ndarray
        Data read from the text file. If `usemask` is True, this is a
        masked array.

    See Also
    --------
    numpy.loadtxt : equivalent function when no data is missing.

    Notes
    -----
    * When spaces are used as delimiters, or when no delimiter has been given
      as input, there should not be any missing data between two fields.
    * When the variables are named (either by a flexible dtype or with `names`,
      there must not be any header in the file (else a ValueError
      exception is raised).
    * Individual values are not stripped of spaces by default.
      When using a custom converter, make sure the function does remove spaces.

    Examples
    ---------
    >>> from StringIO import StringIO
    >>> import numpy as np

    Comma delimited file with mixed dtype

    >>> s = StringIO("1,1.3,abcde")
    >>> data = np.genfromtxt(s, dtype=[('myint','i8'),('myfloat','f8'),
    ... ('mystring','S5')], delimiter=",")
    >>> data
    array((1, 1.3, 'abcde'),
          dtype=[('myint', '<i8'), ('myfloat', '<f8'), ('mystring', '|S5')])

    Using dtype = None

    >>> s.seek(0) # needed for StringIO example only
    >>> data = np.genfromtxt(s, dtype=None,
    ... names = ['myint','myfloat','mystring'], delimiter=",")
    >>> data
    array((1, 1.3, 'abcde'),
          dtype=[('myint', '<i8'), ('myfloat', '<f8'), ('mystring', '|S5')])

    Specifying dtype and names

    >>> s.seek(0)
    >>> data = np.genfromtxt(s, dtype="i8,f8,S5",
    ... names=['myint','myfloat','mystring'], delimiter=",")
    >>> data
    array((1, 1.3, 'abcde'),
          dtype=[('myint', '<i8'), ('myfloat', '<f8'), ('mystring', '|S5')])

    An example with fixed-width columns

    >>> s = StringIO("11.3abcde")
    >>> data = np.genfromtxt(s, dtype=None, names=['intvar','fltvar','strvar'],
    ...     delimiter=[1,3,5])
    >>> data
    array((1, 1.3, 'abcde'),
          dtype=[('intvar', '<i8'), ('fltvar', '<f8'), ('strvar', '|S5')])

    """
    # Py3 data conversions to bytes, for convenience
    comments = asbytes(comments)
    if isinstance(delimiter, unicode):
        delimiter = asbytes(delimiter)
    if isinstance(missing, unicode):
        missing = asbytes(missing)
    if isinstance(missing_values, (unicode, list, tuple)):
        missing_values = asbytes_nested(missing_values)

    #
    if usemask:
        from numpy.ma import MaskedArray, make_mask_descr
    # Check the input dictionary of converters
    user_converters = converters or {}
    if not isinstance(user_converters, dict):
        errmsg = "The input argument 'converter' should be a valid dictionary "\
            "(got '%s' instead)"
        raise TypeError(errmsg % type(user_converters))

    # Initialize the filehandle, the LineSplitter and the NameValidator
    if isinstance(fname, basestring):
        fhd = np.lib._datasource.open(fname)
    elif not hasattr(fname, 'read'):
        raise TypeError("The input should be a string or a filehandle. "\
                        "(got %s instead)" % type(fname))
    else:
        fhd = fname
    split_line = LineSplitter(delimiter=delimiter, comments=comments,
                              autostrip=autostrip)._handyman
    validate_names = NameValidator(excludelist=excludelist,
                                   deletechars=deletechars,
                                   case_sensitive=case_sensitive)

    # Get the first valid lines after the first skiprows ones ..
    if skiprows:
        warnings.warn("The use of `skiprows` is deprecated.\n"\
                      "Please use `skip_header` instead.",
                      DeprecationWarning)
        skip_header = skiprows
    # Skip the first `skip_header` rows
    for i in xrange(skip_header):
        fhd.readline()
    # Keep on until we find the first valid values
    first_values = None
    while not first_values:
        first_line = fhd.readline()
        if not first_line:
            raise IOError('End-of-file reached before encountering data.')
        if names is True:
            if comments in first_line:
                first_line = asbytes('').join(first_line.split(comments)[1:])
        first_values = split_line(first_line)
    # Should we take the first values as names ?
    if names is True:
        fval = first_values[0].strip()
        if fval in comments:
            del first_values[0]

    # Check the columns to use
    if usecols is not None:
        try:
            usecols = [_.strip() for _ in usecols.split(",")]
        except AttributeError:
            try:
                usecols = list(usecols)
            except TypeError:
                usecols = [usecols, ]
    nbcols = len(usecols or first_values)

    # Check the names and overwrite the dtype.names if needed
    if names is True:
        names = validate_names([_bytes_to_name(_.strip())
                                for _ in first_values])
        first_line = asbytes('')
    elif _is_string_like(names):
        names = validate_names([_.strip() for _ in names.split(',')])
    elif names:
        names = validate_names(names)
    # Get the dtype
    if dtype is not None:
        dtype = easy_dtype(dtype, defaultfmt=defaultfmt, names=names)
        names = dtype.names
    # Make sure the names is a list (for 2.5)
    if names is not None:
        names = list(names)


    if usecols:
        for (i, current) in enumerate(usecols):
            # if usecols is a list of names, convert to a list of indices
            if _is_string_like(current):
                usecols[i] = names.index(current)
            elif current < 0:
                usecols[i] = current + len(first_values)
        # If the dtype is not None, make sure we update it
        if (dtype is not None) and (len(dtype) > nbcols):
            descr = dtype.descr
            dtype = np.dtype([descr[_] for _ in usecols])
            names = list(dtype.names)
        # If `names` is not None, update the names
        elif (names is not None) and (len(names) > nbcols):
            names = [names[_] for _ in usecols]


    # Process the missing values ...............................
    # Rename missing_values for convenience
    user_missing_values = missing_values or ()

    # Define the list of missing_values (one column: one list)
    missing_values = [list([asbytes('')]) for _ in range(nbcols)]

    # We have a dictionary: process it field by field
    if isinstance(user_missing_values, dict):
        # Loop on the items
        for (key, val) in user_missing_values.items():
            # Is the key a string ?
            if _is_string_like(key):
                try:
                    # Transform it into an integer
                    key = names.index(key)
                except ValueError:
                    # We couldn't find it: the name must have been dropped, then
                    continue
            # Redefine the key as needed if it's a column number
            if usecols:
                try:
                    key = usecols.index(key)
                except ValueError:
                    pass
            # Transform the value as a list of string
            if isinstance(val, (list, tuple)):
                val = [str(_) for _ in val]
            else:
                val = [str(val), ]
            # Add the value(s) to the current list of missing
            if key is None:
                # None acts as default
                for miss in missing_values:
                    miss.extend(val)
            else:
                missing_values[key].extend(val)
    # We have a sequence : each item matches a column
    elif isinstance(user_missing_values, (list, tuple)):
        for (value, entry) in zip(user_missing_values, missing_values):
            value = str(value)
            if value not in entry:
                entry.append(value)
    # We have a string : apply it to all entries
    elif isinstance(user_missing_values, bytes):
        user_value = user_missing_values.split(asbytes(","))
        for entry in missing_values:
            entry.extend(user_value)
    # We have something else: apply it to all entries
    else:
        for entry in missing_values:
            entry.extend([str(user_missing_values)])

    # Process the deprecated `missing`
    if missing != asbytes(''):
        warnings.warn("The use of `missing` is deprecated.\n"\
                      "Please use `missing_values` instead.",
                      DeprecationWarning)
        values = [str(_) for _ in missing.split(asbytes(","))]
        for entry in missing_values:
            entry.extend(values)

    # Process the filling_values ...............................
    # Rename the input for convenience
    user_filling_values = filling_values or []
    # Define the default
    filling_values = [None] * nbcols
    # We have a dictionary : update each entry individually
    if isinstance(user_filling_values, dict):
        for (key, val) in user_filling_values.items():
            if _is_string_like(key):
                try:
                    # Transform it into an integer
                    key = names.index(key)
                except ValueError:
                    # We couldn't find it: the name must have been dropped, then
                    continue
            # Redefine the key if it's a column number and usecols is defined
            if usecols:
                try:
                    key = usecols.index(key)
                except ValueError:
                    pass
            # Add the value to the list
            filling_values[key] = val
    # We have a sequence : update on a one-to-one basis
    elif isinstance(user_filling_values, (list, tuple)):
        n = len(user_filling_values)
        if (n <= nbcols):
            filling_values[:n] = user_filling_values
        else:
            filling_values = user_filling_values[:nbcols]
    # We have something else : use it for all entries
    else:
        filling_values = [user_filling_values] * nbcols

    # Initialize the converters ................................
    if dtype is None:
        # Note: we can't use a [...]*nbcols, as we would have 3 times the same
        # ... converter, instead of 3 different converters.
        converters = [StringConverter(None, missing_values=miss, default=fill)
                      for (miss, fill) in zip(missing_values, filling_values)]
    else:
        dtype_flat = flatten_dtype(dtype, flatten_base=True)
        # Initialize the converters
        if len(dtype_flat) > 1:
            # Flexible type : get a converter from each dtype
            zipit = zip(dtype_flat, missing_values, filling_values)
            converters = [StringConverter(dt, locked=True,
                                          missing_values=miss, default=fill)
                           for (dt, miss, fill) in zipit]
        else:
            # Set to a default converter (but w/ different missing values)
            zipit = zip(missing_values, filling_values)
            converters = [StringConverter(dtype, locked=True,
                                          missing_values=miss, default=fill)
                          for (miss, fill) in zipit]
    # Update the converters to use the user-defined ones
    uc_update = []
    for (i, conv) in user_converters.items():
        # If the converter is specified by column names, use the index instead
        if _is_string_like(i):
            try:
                i = names.index(i)
            except ValueError:
                continue
        elif usecols:
            try:
                i = usecols.index(i)
            except ValueError:
                # Unused converter specified
                continue
        converters[i].update(conv, locked=True,
                             default=filling_values[i],
                             missing_values=missing_values[i],)
        uc_update.append((i, conv))
    # Make sure we have the corrected keys in user_converters...
    user_converters.update(uc_update)

    miss_chars = [_.missing_values for _ in converters]


    # Initialize the output lists ...
    # ... rows
    rows = []
    append_to_rows = rows.append
    # ... masks
    if usemask:
        masks = []
        append_to_masks = masks.append
    # ... invalid
    invalid = []
    append_to_invalid = invalid.append

    # Parse each line
    for (i, line) in enumerate(itertools.chain([first_line, ], fhd)):
        values = split_line(line)
        nbvalues = len(values)
        # Skip an empty line
        if nbvalues == 0:
            continue
        # Select only the columns we need
        if usecols:
            try:
                values = [values[_] for _ in usecols]
            except IndexError:
                append_to_invalid((i, nbvalues))
                continue
        elif nbvalues != nbcols:
            append_to_invalid((i, nbvalues))
            continue
        # Store the values
        append_to_rows(tuple(values))
        if usemask:
            append_to_masks(tuple([v.strip() in m
                                   for (v, m) in zip(values, missing_values)]))

    # Strip the last skip_footer data
    if skip_footer > 0:
        rows = rows[:-skip_footer]
        if usemask:
            masks = masks[:-skip_footer]

    # Upgrade the converters (if needed)
    if dtype is None:
        for (i, converter) in enumerate(converters):
            current_column = map(itemgetter(i), rows)
            try:
                converter.iterupgrade(current_column)
            except ConverterLockError:
                errmsg = "Converter #%i is locked and cannot be upgraded: " % i
                current_column = itertools.imap(itemgetter(i), rows)
                for (j, value) in enumerate(current_column):
                    try:
                        converter.upgrade(value)
                    except (ConverterError, ValueError):
                        errmsg += "(occurred line #%i for value '%s')"
                        errmsg %= (j + 1 + skip_header, value)
                        raise ConverterError(errmsg)

    # Check that we don't have invalid values
    if len(invalid) > 0:
        nbrows = len(rows)
        # Construct the error message
        template = "    Line #%%i (got %%i columns instead of %i)" % nbcols
        if skip_footer > 0:
            nbrows -= skip_footer
            errmsg = [template % (i + skip_header + 1, nb)
                      for (i, nb) in invalid if i < nbrows]
        else:
            errmsg = [template % (i + skip_header + 1, nb)
                      for (i, nb) in invalid]
        if len(errmsg):
            errmsg.insert(0, "Some errors were detected !")
            errmsg = "\n".join(errmsg)
            # Raise an exception ?
            if invalid_raise:
                raise ValueError(errmsg)
            # Issue a warning ?
            else:
                warnings.warn(errmsg, ConversionWarning)

    # Convert each value according to the converter:
    # We want to modify the list in place to avoid creating a new one...
#    if loose:
#        conversionfuncs = [conv._loose_call for conv in converters]
#    else:
#        conversionfuncs = [conv._strict_call for conv in converters]
#    for (i, vals) in enumerate(rows):
#        rows[i] = tuple([convert(val)
#                         for (convert, val) in zip(conversionfuncs, vals)])
    if loose:
        rows = zip(*[map(converter._loose_call, map(itemgetter(i), rows))
                     for (i, converter) in enumerate(converters)])
    else:
        rows = zip(*[map(converter._strict_call, map(itemgetter(i), rows))
                     for (i, converter) in enumerate(converters)])
    # Reset the dtype
    data = rows
    if dtype is None:
        # Get the dtypes from the types of the converters
        column_types = [conv.type for conv in converters]
        # Find the columns with strings...
        strcolidx = [i for (i, v) in enumerate(column_types)
                     if v in (type('S'), np.string_)]
        # ... and take the largest number of chars.
        for i in strcolidx:
            column_types[i] = "|S%i" % max(len(row[i]) for row in data)
        #
        if names is None:
            # If the dtype is uniform, don't define names, else use ''
            base = set([c.type for c in converters if c._checked])
            if len(base) == 1:
                (ddtype, mdtype) = (list(base)[0], np.bool)
            else:
                ddtype = [(defaultfmt % i, dt)
                          for (i, dt) in enumerate(column_types)]
                if usemask:
                    mdtype = [(defaultfmt % i, np.bool)
                              for (i, dt) in enumerate(column_types)]
        else:
            ddtype = zip(names, column_types)
            mdtype = zip(names, [np.bool] * len(column_types))
        output = np.array(data, dtype=ddtype)
        if usemask:
            outputmask = np.array(masks, dtype=mdtype)
    else:
        # Overwrite the initial dtype names if needed
        if names and dtype.names:
            dtype.names = names
        # Case 1. We have a structured type
        if len(dtype_flat) > 1:
            # Nested dtype, eg  [('a', int), ('b', [('b0', int), ('b1', 'f4')])]
            # First, create the array using a flattened dtype:
            # [('a', int), ('b1', int), ('b2', float)]
            # Then, view the array using the specified dtype.
            if 'O' in (_.char for _ in dtype_flat):
                if has_nested_fields(dtype):
                    errmsg = "Nested fields involving objects "\
                             "are not supported..."
                    raise NotImplementedError(errmsg)
                else:
                    output = np.array(data, dtype=dtype)
            else:
                rows = np.array(data, dtype=[('', _) for _ in dtype_flat])
                output = rows.view(dtype)
            # Now, process the rowmasks the same way
            if usemask:
                rowmasks = np.array(masks,
                                    dtype=np.dtype([('', np.bool)
                                    for t in dtype_flat]))
                # Construct the new dtype
                mdtype = make_mask_descr(dtype)
                outputmask = rowmasks.view(mdtype)
        # Case #2. We have a basic dtype
        else:
            # We used some user-defined converters
            if user_converters:
                ishomogeneous = True
                descr = []
                for (i, ttype) in enumerate([conv.type for conv in converters]):
                    # Keep the dtype of the current converter
                    if i in user_converters:
                        ishomogeneous &= (ttype == dtype.type)
                        if ttype == np.string_:
                            ttype = "|S%i" % max(len(row[i]) for row in data)
                        descr.append(('', ttype))
                    else:
                        descr.append(('', dtype))
                # So we changed the dtype ?
                if not ishomogeneous:
                    # We have more than one field
                    if len(descr) > 1:
                        dtype = np.dtype(descr)
                    # We have only one field: drop the name if not needed.
                    else:
                        dtype = np.dtype(ttype)
            #
            output = np.array(data, dtype)
            if usemask:
                if dtype.names:
                    mdtype = [(_, np.bool) for _ in dtype.names]
                else:
                    mdtype = np.bool
                outputmask = np.array(masks, dtype=mdtype)
    # Try to take care of the missing data we missed
    names = output.dtype.names
    if usemask and names:
        for (name, conv) in zip(names or (), converters):
            missing_values = [conv(_) for _ in conv.missing_values
                              if _ != asbytes('')]
            for mval in missing_values:
                outputmask[name] |= (output[name] == mval)
    # Construct the final array
    if usemask:
        output = output.view(MaskedArray)
        output._mask = outputmask
    if unpack:
        return output.squeeze().T
    return output.squeeze()



def ndfromtxt(fname, **kwargs):
    """
    Load ASCII data stored in a file and return it as a single array.

    Complete description of all the optional input parameters is available in
    the docstring of the `genfromtxt` function.

    See Also
    --------
    numpy.genfromtxt : generic function.

    """
    kwargs['usemask'] = False
    return genfromtxt(fname, **kwargs)


def mafromtxt(fname, **kwargs):
    """
    Load ASCII data stored in a text file and return a masked array.

    For a complete description of all the input parameters, see `genfromtxt`.

    See Also
    --------
    numpy.genfromtxt : generic function to load ASCII data.

    """
    kwargs['usemask'] = True
    return genfromtxt(fname, **kwargs)


def recfromtxt(fname, **kwargs):
    """
    Load ASCII data from a file and return it in a record array.

    If ``usemask=False`` a standard `recarray` is returned,
    if ``usemask=True`` a MaskedRecords array is returned.

    Complete description of all the optional input parameters is available in
    the docstring of the `genfromtxt` function.

    See Also
    --------
    numpy.genfromtxt : generic function

    Notes
    -----
    By default, `dtype` is None, which means that the data-type of the output
    array will be determined from the data.

    """
    kwargs.update(dtype=kwargs.get('dtype', None))
    usemask = kwargs.get('usemask', False)
    output = genfromtxt(fname, **kwargs)
    if usemask:
        from numpy.ma.mrecords import MaskedRecords
        output = output.view(MaskedRecords)
    else:
        output = output.view(np.recarray)
    return output


def recfromcsv(fname, **kwargs):
    """
    Load ASCII data stored in a comma-separated file.

    The returned array is a record array (if ``usemask=False``, see
    `recarray`) or a masked record array (if ``usemask=True``,
    see `ma.mrecords.MaskedRecords`).

    For a complete description of all the input parameters, see `genfromtxt`.

    See Also
    --------
    numpy.genfromtxt : generic function to load ASCII data.

    """
    case_sensitive = kwargs.get('case_sensitive', "lower") or "lower"
    names = kwargs.get('names', True)
    if names is None:
        names = True
    kwargs.update(dtype=kwargs.get('update', None),
                  delimiter=kwargs.get('delimiter', ",") or ",",
                  names=names,
                  case_sensitive=case_sensitive)
    usemask = kwargs.get("usemask", False)
    output = genfromtxt(fname, **kwargs)
    if usemask:
        from numpy.ma.mrecords import MaskedRecords
        output = output.view(MaskedRecords)
    else:
        output = output.view(np.recarray)
    return output


#!/usr/bin/env python
"""
numpyfilter.py INPUTFILE

Interpret C comments as ReStructuredText, and replace them by the HTML output.
Also, add Doxygen /** and /**< syntax automatically where appropriate.

"""
