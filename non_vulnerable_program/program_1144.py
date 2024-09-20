import os
import genapi

types = ['Generic','Numeric','Integer','SignedInteger','UnsignedInteger',
         'Inexact',
         'Floating', 'ComplexFloating', 'Flexible', 'Character',
         'Bool','Byte','Short','Int', 'Long', 'LongLong', 'UByte', 'UShort',
         'UInt', 'ULong', 'ULongLong', 'Float', 'Double', 'LongDouble',
         'CFloat', 'CDouble', 'CLongDouble', 'Object', 'String', 'Unicode',
         'Void']

h_template = r"""
#ifdef _MULTIARRAYMODULE

static PyTypeObject PyBigArray_Type;
static PyTypeObject PyArray_Type;
static PyTypeObject PyArrayDescr_Type;
static PyTypeObject PyArrayIter_Type;
static PyTypeObject PyArrayMapIter_Type;
static PyTypeObject PyArrayMultiIter_Type;
static int PyArray_NUMUSERTYPES=0;

%s

#else

#if defined(PY_ARRAY_UNIQUE_SYMBOL)
#define PyArray_API PY_ARRAY_UNIQUE_SYMBOL
#endif

#if defined(NO_IMPORT) || defined(NO_IMPORT_ARRAY)
extern void **PyArray_API;
#else
#if defined(PY_ARRAY_UNIQUE_SYMBOL)
void **PyArray_API;
#else
static void **PyArray_API=NULL;
#endif
#endif

#define PyBigArray_Type (*(PyTypeObject *)PyArray_API[0])
#define PyArray_Type (*(PyTypeObject *)PyArray_API[1])
#define PyArrayDescr_Type (*(PyTypeObject *)PyArray_API[2])
#define PyArrayIter_Type (*(PyTypeObject *)PyArray_API[3])
#define PyArrayMultiIter_Type (*(PyTypeObject *)PyArray_API[4])
#define PyArray_NUMUSERTYPES (*(int *)PyArray_API[5])

%s

#if !defined(NO_IMPORT_ARRAY) && !defined(NO_IMPORT)
static int
