from base import Base
from c_code import CCode

Base.register(

    CCode('#include "Python.h"', provides='Python.h'),

    CCode('''\
#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"
''', provides='arrayobject.h'),

    CCode('''\
