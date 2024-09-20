import Numeric

class Vector(bytecodecompiler.Type_Descriptor):
    cxxtype = 'PyArrayObject*'
    prerequisites = bytecodecompiler.Type_Descriptor.prerequisites+\
                   ['#include "Numeric/arrayobject.h"',
                    'static PyObject* PyArray_AsPyObject(PyArrayObject* A) { PyObject* X = '
                    'reinterpret_cast<PyObject*>(A);'
                    'std::cerr << "Here in cast" << std::endl;'
                    'Py_XINCREF(X); return X;}']
    dims = 1
    def check(self,s):
        return "PyArray_Check(%s) /* && dims==%d && typecode==%s */"%(s,self.dims,self.typecode)
    def inbound(self,s):
        return "(PyArrayObject*)(%s)"%s
    def outbound(self,s):
        return "(PyObject*)(%s)"%s

class IntegerVector(Vector):
    typecode = 'l'

typedefs = {
    IntType: Integer(),
    FloatType: Double(),
    StringType: String(),
    (Numeric.ArrayType,1,'l'): IntegerVector(),
    }


##################################################################
#                      FUNCTION LOOKUP_TYPE                      #
##################################################################
def lookup_type(x):
    T = type(x)
    try:
        return typedefs[T]
    except:
        return typedefs[(T,len(x.shape),x.typecode())]

##################################################################
#                        class ACCELERATE                        #
##################################################################
class accelerate:
    
    def __init__(self, function, *args, **kw):
        assert type(function) == FunctionType
        self.function = function
        self.module = inspect.getmodule(function)
        if self.module == None:
            import __main__
            self.module = __main__
        self.__call_map = {}
        return

    def __call__(self,*args):
        # Figure out type info -- Do as tuple so its hashable
        signature = tuple( map(lookup_type,args) )

        # If we know the function, call it
        try:
            return self.__call_map[signature](*args)
        except:
            fast = self.singleton(signature)
            self.__call_map[signature] = fast
            return fast(*args)

    def signature(self,*args):
        # Figure out type info -- Do as tuple so its hashable
        signature = tuple( map(lookup_type,args) )
        return self.singleton(signature)


    def singleton(self,signature):
        identifier = self.identifier(signature)
        
        # Generate a new function, then call it
        f = self.function

        # See if we have an accelerated version of module
        try:
            accelerated_module = __import__(self.module.__name__+'_weave')
            fast = getattr(accelerated_module,identifier)
            return fast
        except:
            accelerated_module = None

        P = self.accelerate(signature,identifier)

        E = weave.ext_tools.ext_module(self.module.__name__+'_weave')
        E.add_function(P)
        E.generate_file()
        weave.build_tools.build_extension(self.module.__name__+'_weave.cpp',verbose=2)

        if accelerated_module:
            accelerated_module = reload(accelerated_module)
        else:
            accelerated_module = __import__(self.module.__name__+'_weave')

        fast = getattr(accelerated_module,identifier)
        return fast

    def identifier(self,signature):
        # Build an MD5 checksum
        f = self.function
        co = f.func_code
        identifier = str(signature)+\
                     str(co.co_consts)+\
                     str(co.co_varnames)+\
                     co.co_code
        return 'F'+md5.md5(identifier).hexdigest()
        
    def accelerate(self,signature,identifier):
        P = Python2CXX(self.function,signature,name=identifier)
        return P

    def code(self,*args):
        signature = tuple( map(lookup_type,args) )
        ident = self.function.__name__
        return self.accelerate(signature,ident).function_code()
        

##################################################################
#                        CLASS PYTHON2CXX                        #
##################################################################
class Python2CXX(bytecodecompiler.CXXCoder):
    def typedef_by_value(self,v):
        T = lookup_type(v)
        if T not in self.used: self.used.append(T)
        return T

    def __init__(self,f,signature,name=None):
        # Make sure function is a function
        import types
        assert type(f) == FunctionType
        # and check the input type signature
        assert reduce(lambda x,y: x and y,
                      map(lambda x: isinstance(x,bytecodecompiler.Type_Descriptor),
                          signature),
                      1),'%s not all type objects'%signature
        self.arg_specs = []
        self.customize = weave.base_info.custom_info()

        bytecodecompiler.CXXCoder.__init__(self,f,signature,name)
        return

    def function_code(self):
        return self.wrapped_code()

    def python_function_definition_code(self):
        return '{ "%s", wrapper_%s, METH_VARARGS, %s },\n'%(
            self.name,
            self.name,
            CStr(self.function.__doc__))


#**************************************************************************#
#* FILE   **************    bytecodecompiler.py    ************************#
#************************************************************************ **#
#* Author: Patrick Miller February  9 2002                                *#
#* Copyright (C) 2002 Patrick J. Miller                                   *#
#**************************************************************************#
#*  *#
#**************************************************************************#
