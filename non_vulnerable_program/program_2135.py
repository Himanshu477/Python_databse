import os
import genapi

types = ['Generic','Number','Integer','SignedInteger','UnsignedInteger',
         'Inexact',
         'Floating', 'ComplexFloating', 'Flexible', 'Character',
         'Byte','Short','Int', 'Long', 'LongLong', 'UByte', 'UShort',
         'UInt', 'ULong', 'ULongLong', 'Float', 'Double', 'LongDouble',
         'CFloat', 'CDouble', 'CLongDouble', 'Object', 'String', 'Unicode',
         'Void']

h_template = r"""
#ifdef _MULTIARRAYMODULE

typedef struct {
        PyObject_HEAD
        npy_bool obval;
} PyBoolScalarObject;


static unsigned int PyArray_GetNDArrayCVersion (void);
static PyTypeObject PyBigArray_Type;
static PyTypeObject PyArray_Type;
static PyTypeObject PyArrayDescr_Type;
static PyTypeObject PyArrayFlags_Type;
static PyTypeObject PyArrayIter_Type;
static PyTypeObject PyArrayMapIter_Type;
static PyTypeObject PyArrayMultiIter_Type;
static int NPY_NUMUSERTYPES=0;
static PyTypeObject PyBoolArrType_Type;
static PyBoolScalarObject _PyArrayScalar_BoolValues[2];

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

#define PyArray_GetNDArrayCVersion (*(unsigned int (*)(void)) PyArray_API[0])
#define PyBigArray_Type (*(PyTypeObject *)PyArray_API[1])
#define PyArray_Type (*(PyTypeObject *)PyArray_API[2])
#define PyArrayDescr_Type (*(PyTypeObject *)PyArray_API[3])
#define PyArrayFlags_Type (*(PyTypeObject *)PyArray_API[4])
#define PyArrayIter_Type (*(PyTypeObject *)PyArray_API[5])
#define PyArrayMultiIter_Type (*(PyTypeObject *)PyArray_API[6])
#define NPY_NUMUSERTYPES (*(int *)PyArray_API[7])
#define PyBoolArrType_Type (*(PyTypeObject *)PyArray_API[8])
#define _PyArrayScalar_BoolValues ((PyBoolScalarObject *)PyArray_API[9])

%s

#if !defined(NO_IMPORT_ARRAY) && !defined(NO_IMPORT)
static int
_import_array(void)
{
  PyObject *numpy = PyImport_ImportModule("numpy.core.multiarray");
  PyObject *c_api = NULL;
  if (numpy == NULL) return -1;
  c_api = PyObject_GetAttrString(numpy, "_ARRAY_API");
  if (c_api == NULL) {Py_DECREF(numpy); return -1;}
  if (PyCObject_Check(c_api)) {
      PyArray_API = (void **)PyCObject_AsVoidPtr(c_api);
  }
  Py_DECREF(c_api);
  Py_DECREF(numpy);
  if (PyArray_API == NULL) return -1;
  /* Perform runtime check of C API version */
  if (NPY_VERSION != PyArray_GetNDArrayCVersion()) {
    PyErr_Format(PyExc_RuntimeError, "module compiled against "\
        "version %%x of C-API but this version of numpy is %%x", \
        (int) NPY_VERSION, (int) PyArray_GetNDArrayCVersion());
    return -1;
  }
  return 0;
}

#define import_array() {if (_import_array() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import"); return; } }

#define import_array1(ret) {if (_import_array() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import"); return ret; } }

#define import_array2(msg, ret) {if (_import_array() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, msg); return ret; } }

#endif

#endif
"""


c_template = r"""
/* These pointers will be stored in the C-object for use in other
    extension modules
*/

void *PyArray_API[] = {
        (void *) PyArray_GetNDArrayCVersion,
        (void *) &PyBigArray_Type,
        (void *) &PyArray_Type,
        (void *) &PyArrayDescr_Type,
        (void *) &PyArrayFlags_Type,
        (void *) &PyArrayIter_Type,
        (void *) &PyArrayMultiIter_Type,
        (int *) &NPY_NUMUSERTYPES,
        (void *) &PyBoolArrType_Type,
        (void *) &_PyArrayScalar_BoolValues,
%s
};
"""

c_api_header = """
===========
Numpy C-API
===========
"""

def generate_api(output_dir, force=False):
    basename = 'multiarray_api'

    h_file = os.path.join(output_dir, '__%s.h' % basename)
    c_file = os.path.join(output_dir, '__%s.c' % basename)
    d_file = os.path.join(output_dir, '%s.txt' % basename)
    targets = (h_file, c_file, d_file)
    sources = ['numpy_api_order.txt']

    if (not force and not genapi.should_rebuild(targets, sources + [__file__])):
        return targets
    else:
        do_generate_api(targets, sources)

    return targets

def do_generate_api(targets, sources):
    header_file = targets[0]
    c_file = targets[1]
    doc_file = targets[2]

    numpyapi_list = genapi.get_api_functions('NUMPY_API', sources[0])

    # API fixes for __arrayobject_api.h
    fixed = 10
    numtypes = len(types) + fixed

    module_list = []
    extension_list = []
    init_list = []

    # setup types
    for k, atype in enumerate(types):
        num = fixed + k
        astr = "        (void *) &Py%sArrType_Type," % types[k]
        init_list.append(astr)
        astr = "static PyTypeObject Py%sArrType_Type;" % types[k]
        module_list.append(astr)
        astr = "#define Py%sArrType_Type (*(PyTypeObject *)PyArray_API[%d])" % \
               (types[k], num)
        extension_list.append(astr)

    # set up object API
    genapi.add_api_list(numtypes, 'PyArray_API', numpyapi_list,
                        module_list, extension_list, init_list)

    # Write to header
    fid = open(header_file, 'w')
    s = h_template % ('\n'.join(module_list), '\n'.join(extension_list))
    fid.write(s)
    fid.close()

    # Write to c-code
    fid = open(c_file, 'w')
    s = c_template % '\n'.join(init_list)
    fid.write(s)
    fid.close()

    # write to documentation
    fid = open(doc_file, 'w')
    fid.write(c_api_header)
    for func in numpyapi_list:
        fid.write(func.to_ReST())
        fid.write('\n\n')
    fid.close()

    return targets


# Docstrings for generated ufuncs

docdict = {}

def get(name):
    return docdict.get(name)

def add_newdoc(place, name, doc):
    docdict['.'.join((place, name))] = doc


add_newdoc('numpy.core.umath', 'absolute',
    """
    Takes |x| elementwise.

    """)

add_newdoc('numpy.core.umath', 'add',
    """
    Adds the arguments elementwise.

    """)

add_newdoc('numpy.core.umath', 'arccos',
    """
    Inverse cosine elementwise.

    """)

add_newdoc('numpy.core.umath', 'arccosh',
    """
    Inverse hyperbolic cosine elementwise.

    """)

add_newdoc('numpy.core.umath', 'arcsin',
    """
    Inverse sine elementwise.

    """)

add_newdoc('numpy.core.umath', 'arcsinh',
    """
    Inverse hyperbolic sine elementwise.

    """)

add_newdoc('numpy.core.umath', 'arctan',
    """
    Inverse tangent elementwise.

    """)

add_newdoc('numpy.core.umath', 'arctan2',
    """
    A safe and correct arctan(x1/x2)

    """)

add_newdoc('numpy.core.umath', 'arctanh',
    """
    Inverse hyperbolic tangent elementwise.

    """)

add_newdoc('numpy.core.umath', 'bitwise_and',
    """
    Computes x1 & x2 elementwise.

    """)

add_newdoc('numpy.core.umath', 'bitwise_or',
    """
    Computes x1 | x2 elementwise.

    """)

add_newdoc('numpy.core.umath', 'bitwise_xor',
    """
    Computes x1 ^ x2 elementwise.

    """)

add_newdoc('numpy.core.umath', 'ceil',
    """
    Elementwise smallest integer >= x.

    """)

add_newdoc('numpy.core.umath', 'conjugate',
    """
    Takes the conjugate of x elementwise.

    """)

add_newdoc('numpy.core.umath', 'cos',
    """
    Cosine elementwise.

    """)

add_newdoc('numpy.core.umath', 'cosh',
    """
    Hyperbolic cosine elementwise.

    """)

add_newdoc('numpy.core.umath', 'degrees',
    """
    Converts angle from radians to degrees

    """)

add_newdoc('numpy.core.umath', 'divide',
    """
    Divides the arguments elementwise.

    """)

add_newdoc('numpy.core.umath', 'equal',
    """
    Returns elementwise x1 == x2 in a bool array

    """)

add_newdoc('numpy.core.umath', 'exp',
    """
    e**x elementwise.

    """)

add_newdoc('numpy.core.umath', 'expm1',
    """
    e**x-1 elementwise.

    """)

add_newdoc('numpy.core.umath', 'fabs',
    """
    Absolute values.

    """)

add_newdoc('numpy.core.umath', 'floor',
    """
    Elementwise largest integer <= x

    """)

add_newdoc('numpy.core.umath', 'floor_divide',
    """
    Floor divides the arguments elementwise.

    """)

add_newdoc('numpy.core.umath', 'fmod',
    """
    Computes (C-like) x1 % x2 elementwise.

    """)

add_newdoc('numpy.core.umath', 'greater',
    """
    Returns elementwise x1 > x2 in a bool array.

    """)

add_newdoc('numpy.core.umath', 'greater_equal',
    """
    Returns elementwise x1 >= x2 in a bool array.

    """)

add_newdoc('numpy.core.umath', 'hypot',
    """
    sqrt(x1**2 + x2**2) elementwise

    """)

add_newdoc('numpy.core.umath', 'invert',
    """
    Computes ~x (bit inversion) elementwise.

    """)

add_newdoc('numpy.core.umath', 'isfinite',
    """
    Returns True where x is finite

    """)

add_newdoc('numpy.core.umath', 'isinf',
    """
    Returns True where x is +inf or -inf

    """)

add_newdoc('numpy.core.umath', 'isnan',
    """
    Returns True where x is Not-A-Number

    """)

add_newdoc('numpy.core.umath', 'left_shift',
    """
    Computes x1 << x2 (x1 shifted to left by x2 bits) elementwise.

    """)

add_newdoc('numpy.core.umath', 'less',
    """
    Returns elementwise x1 < x2 in a bool array.

    """)

add_newdoc('numpy.core.umath', 'less_equal',
    """
    Returns elementwise x1 <= x2 in a bool array

    """)

add_newdoc('numpy.core.umath', 'log',
    """
    Logarithm base e elementwise.

    """)

add_newdoc('numpy.core.umath', 'log10',
    """
    Logarithm base 10 elementwise.

    """)

add_newdoc('numpy.core.umath', 'log1p',
    """
    log(1+x) to base e elementwise.

    """)

add_newdoc('numpy.core.umath', 'logical_and',
    """
    Returns x1 and x2 elementwise.

    """)

add_newdoc('numpy.core.umath', 'logical_not',
    """
    Returns not x elementwise.

    """)

add_newdoc('numpy.core.umath', 'logical_or',
    """
    Returns x1 or x2 elementwise.

    """)

add_newdoc('numpy.core.umath', 'logical_xor',
    """
    Returns x1 xor x2 elementwise.

    """)

add_newdoc('numpy.core.umath', 'maximum',
    """
    Returns maximum (if x1 > x2: x1;  else: x2) elementwise.

    """)

add_newdoc('numpy.core.umath', 'minimum',
    """
    Returns minimum (if x1 < x2: x1;  else: x2) elementwise

    """)

add_newdoc('numpy.core.umath', 'modf',
    """
    Breaks x into fractional (y1) and integral (y2) parts.

    Each output has the same sign as the input.

    """)

add_newdoc('numpy.core.umath', 'multiply',
    """
    Multiplies the arguments elementwise.

    """)

add_newdoc('numpy.core.umath', 'negative',
    """
    Determines -x elementwise

    """)

add_newdoc('numpy.core.umath', 'not_equal',
    """
    Returns elementwise x1 |= x2

    """)

add_newdoc('numpy.core.umath', 'ones_like',
    """
    Returns an array of ones of the shape and typecode of x.

    """)

add_newdoc('numpy.core.umath', 'power',
    """
    Computes x1**x2 elementwise.

    """)

add_newdoc('numpy.core.umath', 'radians',
    """
    Converts angle from degrees to radians

    """)

add_newdoc('numpy.core.umath', 'reciprocal',
    """
    Compute 1/x

    """)

add_newdoc('numpy.core.umath', 'remainder',
    """
    Computes x1-n*x2 where n is floor(x1 / x2)

    """)

add_newdoc('numpy.core.umath', 'right_shift',
    """
    Computes x1 >> x2 (x1 shifted to right by x2 bits) elementwise.

    """)

add_newdoc('numpy.core.umath', 'rint',
    """
    Round x elementwise to the nearest integer, round halfway cases away from zero

    """)

add_newdoc('numpy.core.umath', 'sign',
    """
    Returns -1 if x < 0 and 0 if x==0 and 1 if x > 0

    """)

add_newdoc('numpy.core.umath', 'signbit',
    """
    Returns True where signbit of x is set (x<0).

    """)

add_newdoc('numpy.core.umath', 'sin',
    """
    Sine elementwise.

    """)

add_newdoc('numpy.core.umath', 'sinh',
    """
    Hyperbolic sine elementwise.

    """)

add_newdoc('numpy.core.umath', 'sqrt',
    """
    Square-root elementwise. For real x, the domain is restricted to x>=0.

    """)

add_newdoc('numpy.core.umath', 'square',
    """
    Compute x**2.

    """)

add_newdoc('numpy.core.umath', 'subtract',
    """
    Subtracts the arguments elementwise.

    """)

add_newdoc('numpy.core.umath', 'tan',
    """
    Tangent elementwise.

    """)

add_newdoc('numpy.core.umath', 'tanh',
    """
    Hyperbolic tangent elementwise.

    """)

add_newdoc('numpy.core.umath', 'true_divide',
    """
    True divides the arguments elementwise.

    """)



# To get sub-modules
