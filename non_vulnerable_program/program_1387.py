    import imp
    svn = imp.load_module('numpy.core.__svn_version__',
                          open(svn_version_file),
                          svn_version_file,
                          ('.py','U',1))
    version += '.'+svn.version


# Borrowed and adapted from numarray

"""numerictypes: Define the numeric type objects

This module is designed so 'from numerictypes import *' is safe.
Exported symbols include:

  Dictionary with all registered number types (including aliases):
    typeDict

  Type objects (not all will be available, depends on platform):
      see variable arraytypes for which ones you have

    Bit-width names

    int8 int16 int32 int64 int128
    uint8 uint16 uint32 uint64 uint128
    float16 float32 float64 float96 float128 float256
    complex32 complex64 complex128 complex192 complex256 complex512

    c-based names

    bool_

    object_

    void, str_, unicode_

    byte, ubyte,
    short, ushort
    intc, uintc,
    intp, uintp,
    int_, uint,
    longlong, ulonglong,

    single, csingle,
    float_, complex_,
    longfloat, clongfloat,

    As part of the type-hierarchy:    xx -- is bit-width

     generic
       bool_
       number
         integer
           signedinteger   (intxx)
             byte
             short
             intc
             intp           int0
             int_
             longlong
           unsignedinteger  (uintxx)
             ubyte
             ushort
             uintc
             uintp          uint0
             uint_
             ulonglong
         floating           (floatxx)
             single
             float_  (double)
             longfloat
         complexfloating    (complexxx)
             csingle
             complex_ (cfloat, cdouble)
             clongfloat

       flexible
         character
           str_     (string)
           unicode_
         void

       object_

$Id: numerictypes.py,v 1.17 2005/09/09 22:20:06 teoliphant Exp $
"""

# we add more at the bottom
__all__ = ['typeDict', 'typeNA', 'arraytypes', 'ScalarType', 'obj2dtype', 'cast', 'nbytes', 'dtype2char']

