from types import InstanceType,FunctionType,IntType,FloatType,StringType,TypeType,XRangeType
import inspect
import md5
import weave
import imp
from bytecodecompiler import CXXCoder,Type_Descriptor,Function_Descriptor

def CStr(s):
    "Hacky way to get legal C string from Python string"
    if s is None: return '""'
    assert type(s) == StringType,"Only None and string allowed"
    r = repr('"'+s) # Better for embedded quotes
    return '"'+r[2:-1]+'"'


##################################################################
#                         CLASS INSTANCE                         #
##################################################################
class Instance(Type_Descriptor):
    cxxtype = 'PyObject*'
    
    def __init__(self,prototype):
    self.prototype    = prototype
    return

    def check(self,s):
        return "PyInstance_Check(%s)"%s

    def inbound(self,s):
        return s

    def outbound(self,s):
        return s,0

    def get_attribute(self,name):
        proto = getattr(self.prototype,name)
        T = lookup_type(proto)
        code = 'tempPY = PyObject_GetAttrString(%%(rhs)s,"%s");\n'%name
        convert = T.inbound('tempPY')
        code += '%%(lhsType)s %%(lhs)s = %s;\n'%convert
        return T,code

    def set_attribute(self,name):
        proto = getattr(self.prototype,name)
        T = lookup_type(proto)
        convert,owned = T.outbound('%(rhs)s')
        code = 'tempPY = %s;'%convert
        if not owned:
            code += ' Py_INCREF(tempPY);'
        code += ' PyObject_SetAttrString(%%(lhs)s,"%s",tempPY);'%name
        code += ' Py_DECREF(tempPY);\n'
        return T,code

##################################################################
#                          CLASS BASIC                           #
##################################################################
class Basic(Type_Descriptor):
    owned = 1
    def check(self,s):
        return "%s(%s)"%(self.checker,s)
    def inbound(self,s):
        return "%s(%s)"%(self.inbounder,s)
    def outbound(self,s):
        return "%s(%s)"%(self.outbounder,s),self.owned

class Basic_Number(Basic):
    def literalizer(self,s):
        return str(s)
    def binop(self,symbol,a,b):
        assert symbol in ['+','-','*','/'],symbol
        return '%s %s %s'%(a,symbol,b),self

class Integer(Basic_Number):
    cxxtype = "long"
    checker = "PyInt_Check"
    inbounder = "PyInt_AsLong"
    outbounder = "PyInt_FromLong"

class Double(Basic_Number):
    cxxtype = "double"
    checker = "PyFloat_Check"
    inbounder = "PyFloat_AsDouble"
    outbounder = "PyFloat_FromDouble"

class String(Basic):
    cxxtype = "char*"
    checker = "PyString_Check"
    inbounder = "PyString_AsString"
    outbounder = "PyString_FromString"

    def literalizer(self,s):
        return CStr(s)

# -----------------------------------------------
# Singletonize the type names
# -----------------------------------------------
Integer = Integer()
Double = Double()
String = String()

