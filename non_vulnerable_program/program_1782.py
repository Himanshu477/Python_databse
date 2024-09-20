from parser.api import *
from wrapper_base import *

class PythonWrapperModule(WrapperBase):

    main_template = '''\
#ifdef __cplusplus
extern \"C\" {
#endif
#include "Python.h"

#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

%(header_list)s

%(typedef_list)s

%(extern_list)s

%(c_code_list)s

%(capi_code_list)s

%(objdecl_list)s

static PyObject *f2py_module;

static PyMethodDef f2py_module_methods[] = {
  %(module_method_list)s
  {NULL,NULL,0,NULL}
};

PyMODINIT_FUNC init%(modulename)s(void) {
  f2py_module = Py_InitModule("%(modulename)s", f2py_module_methods);
