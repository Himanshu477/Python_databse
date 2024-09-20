import base_info

# this code will not build with msvc...
scalar_support_code = \
"""
// conversion routines

template<class T> 
static T convert_to_scalar(PyObject* py_obj,char* name)
{
    //never used.
    return (T) 0;
}
template<>
static int convert_to_scalar<int>(PyObject* py_obj,char* name)
{
    if (!py_obj || !PyInt_Check(py_obj))
        handle_conversion_error(py_obj,"int", name);
    return (int) PyInt_AsLong(py_obj);
}

template<>
static long convert_to_scalar<long>(PyObject* py_obj,char* name)
{
    if (!py_obj || !PyLong_Check(py_obj))
        handle_conversion_error(py_obj,"long", name);
    return (long) PyLong_AsLong(py_obj);
}

template<> 
static double convert_to_scalar<double>(PyObject* py_obj,char* name)
{
    if (!py_obj || !PyFloat_Check(py_obj))
        handle_conversion_error(py_obj,"float", name);
    return PyFloat_AsDouble(py_obj);
}

template<> 
static float convert_to_scalar<float>(PyObject* py_obj,char* name)
{
    return (float) convert_to_scalar<double>(py_obj,name);
}

// complex not checked.
template<> 
static std::complex<float> convert_to_scalar<std::complex<float> >(PyObject* py_obj,
                                                              char* name)
{
    if (!py_obj || !PyComplex_Check(py_obj))
        handle_conversion_error(py_obj,"complex", name);
    return std::complex<float>((float) PyComplex_RealAsDouble(py_obj),
                               (float) PyComplex_ImagAsDouble(py_obj));    
}
template<> 
static std::complex<double> convert_to_scalar<std::complex<double> >(
                                            PyObject* py_obj,char* name)
{
    if (!py_obj || !PyComplex_Check(py_obj))
        handle_conversion_error(py_obj,"complex", name);
    return std::complex<double>(PyComplex_RealAsDouble(py_obj),
                                PyComplex_ImagAsDouble(py_obj));    
}

/////////////////////////////////
// standard translation routines

template<class T> 
static T py_to_scalar(PyObject* py_obj,char* name)
{
    //never used.
    return (T) 0;
}
template<>
static int py_to_scalar<int>(PyObject* py_obj,char* name)
{
    if (!py_obj || !PyInt_Check(py_obj))
        handle_bad_type(py_obj,"int", name);
    return (int) PyInt_AsLong(py_obj);
}

template<>
static long py_to_scalar<long>(PyObject* py_obj,char* name)
{
    if (!py_obj || !PyLong_Check(py_obj))
        handle_bad_type(py_obj,"long", name);
    return (long) PyLong_AsLong(py_obj);
}

template<> 
static double py_to_scalar<double>(PyObject* py_obj,char* name)
{
    if (!py_obj || !PyFloat_Check(py_obj))
        handle_bad_type(py_obj,"float", name);
    return PyFloat_AsDouble(py_obj);
}

template<> 
static float py_to_scalar<float>(PyObject* py_obj,char* name)
{
    return (float) py_to_scalar<double>(py_obj,name);
}

// complex not checked.
template<> 
static std::complex<float> py_to_scalar<std::complex<float> >(PyObject* py_obj,
                                                              char* name)
{
    if (!py_obj || !PyComplex_Check(py_obj))
        handle_bad_type(py_obj,"complex", name);
    return std::complex<float>((float) PyComplex_RealAsDouble(py_obj),
                               (float) PyComplex_ImagAsDouble(py_obj));    
}
template<> 
static std::complex<double> py_to_scalar<std::complex<double> >(
                                            PyObject* py_obj,char* name)
{
    if (!py_obj || !PyComplex_Check(py_obj))
        handle_bad_type(py_obj,"complex", name);
    return std::complex<double>(PyComplex_RealAsDouble(py_obj),
                                PyComplex_ImagAsDouble(py_obj));    
}
"""    

non_template_scalar_support_code = \
"""

// Conversion Errors

static int convert_to_int(PyObject* py_obj,char* name)
{
    if (!py_obj || !PyInt_Check(py_obj))
        handle_conversion_error(py_obj,"int", name);
    return (int) PyInt_AsLong(py_obj);
}

static long convert_to_long(PyObject* py_obj,char* name)
{
    if (!py_obj || !PyLong_Check(py_obj))
        handle_conversion_error(py_obj,"long", name);
    return (long) PyLong_AsLong(py_obj);
}

static double convert_to_float(PyObject* py_obj,char* name)
{
    if (!py_obj || !PyFloat_Check(py_obj))
        handle_conversion_error(py_obj,"float", name);
    return PyFloat_AsDouble(py_obj);
}

// complex not checked.
static std::complex<double> convert_to_complex(PyObject* py_obj,char* name)
{
    if (!py_obj || !PyComplex_Check(py_obj))
        handle_conversion_error(py_obj,"complex", name);
    return std::complex<double>(PyComplex_RealAsDouble(py_obj),
                                PyComplex_ImagAsDouble(py_obj));    
}

/////////////////////////////////////
// The following functions are used for scalar conversions in msvc
// because it doesn't handle templates as well.

static int py_to_int(PyObject* py_obj,char* name)
{
    if (!py_obj || !PyInt_Check(py_obj))
        handle_bad_type(py_obj,"int", name);
    return (int) PyInt_AsLong(py_obj);
}

static long py_to_long(PyObject* py_obj,char* name)
{
    if (!py_obj || !PyLong_Check(py_obj))
        handle_bad_type(py_obj,"long", name);
    return (long) PyLong_AsLong(py_obj);
}

static double py_to_float(PyObject* py_obj,char* name)
{
    if (!py_obj || !PyFloat_Check(py_obj))
        handle_bad_type(py_obj,"float", name);
    return PyFloat_AsDouble(py_obj);
}

// complex not checked.
static std::complex<double> py_to_complex(PyObject* py_obj,char* name)
{
    if (!py_obj || !PyComplex_Check(py_obj))
        handle_bad_type(py_obj,"complex", name);
    return std::complex<double>(PyComplex_RealAsDouble(py_obj),
                                PyComplex_ImagAsDouble(py_obj));    
}
"""    


