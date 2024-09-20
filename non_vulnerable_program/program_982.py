import scipy.base as nx

class Vector(Type_Descriptor):
    cxxtype = 'PyArrayObject*'
    refcount = 1
    dims = 1
    module_init_code = 'import_array();\n'
    inbounder = "(PyArrayObject*)"
    outbounder = "(PyObject*)"
    owned = 0 # Convertion is by casting!

    prerequisites = Type_Descriptor.prerequisites+\
                   ['#include "scipy/arrayobject.h"']
    dims = 1
    def check(self,s):
        return "PyArray_Check(%s) && ((PyArrayObject*)%s)->nd == %d &&  ((PyArrayObject*)%s)->descr->type_num == %s"%(
            s,s,self.dims,s,self.typecode)

    def inbound(self,s):
        return "%s(%s)"%(self.inbounder,s)
    def outbound(self,s):
        return "%s(%s)"%(self.outbounder,s),self.owned

    def getitem(self,A,v,t):
        assert self.dims == len(v),'Expect dimension %d'%self.dims
        code = '*((%s*)(%s->data'%(self.cxxbase,A)
        for i in range(self.dims):
            # assert that ''t[i]'' is an integer
            code += '+%s*%s->strides[%d]'%(v[i],A,i)
        code += '))'
        return code,self.pybase
    def setitem(self,A,v,t):
        return self.getitem(A,v,t)

class matrix(Vector):
    dims = 2

class IntegerVector(Vector):
    typecode = 'PyArray_INT'
    cxxbase = 'int'
    pybase = Integer

class Integermatrix(matrix):
    typecode = 'PyArray_INT'
    cxxbase = 'int'
    pybase = Integer

class LongVector(Vector):
    typecode = 'PyArray_LONG'
    cxxbase = 'long'
    pybase = Integer

class Longmatrix(matrix):
    typecode = 'PyArray_LONG'
    cxxbase = 'long'
    pybase = Integer

class DoubleVector(Vector):
    typecode = 'PyArray_DOUBLE'
    cxxbase = 'double'
    pybase = Double

class Doublematrix(matrix):
    typecode = 'PyArray_DOUBLE'
    cxxbase = 'double'
    pybase = Double


##################################################################
#                          CLASS XRANGE                          #
##################################################################
class XRange(Type_Descriptor):
    cxxtype = 'XRange'
    prerequisites = ['''
    class XRange {
    public:
    XRange(long aLow, long aHigh, long aStep=1)
    : low(aLow),high(aHigh),step(aStep)
    {
    }
    XRange(long aHigh)
    : low(0),high(aHigh),step(1)
    {
    }
    long low;
    long high;
    long step;
    };''']

# -----------------------------------------------
# Singletonize the type names
# -----------------------------------------------
IntegerVector = IntegerVector()
Integermatrix = Integermatrix()
LongVector = LongVector()
Longmatrix = Longmatrix()
DoubleVector = DoubleVector()
Doublematrix = Doublematrix()
XRange = XRange()


typedefs = {
    IntType: Integer,
    FloatType: Double,
    StringType: String,
    (nx.ArrayType,1,'i'): IntegerVector,
    (nx.ArrayType,2,'i'): Integermatrix,
    (nx.ArrayType,1,'l'): LongVector,
    (nx.ArrayType,2,'l'): Longmatrix,
    (nx.ArrayType,1,'d'): DoubleVector,
    (nx.ArrayType,2,'d'): Doublematrix,
    XRangeType : XRange,
    }

