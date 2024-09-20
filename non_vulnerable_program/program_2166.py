import Fortran

######################################################################

class FortranTestCase(unittest.TestCase):

    def __init__(self, methodName="runTests"):
        unittest.TestCase.__init__(self, methodName)
        self.typeStr  = "double"
        self.typeCode = "d"

    # Test (type* IN_FARRAY2, int DIM1, int DIM2) typemap
    def testSecondElementContiguous(self):
        "Test luSplit function with a Fortran-array"
        print >>sys.stderr, self.typeStr, "... ",
        second = Fortran.__dict__[self.typeStr + "SecondElement"]
        matrix = np.arange(9).reshape(3, 3).astype(self.typeCode)
        self.assertEquals(second(matrix), 3)

    def testSecondElementFortran(self):
        "Test luSplit function with a Fortran-array"
        print >>sys.stderr, self.typeStr, "... ",
        second = Fortran.__dict__[self.typeStr + "SecondElement"]
        matrix = np.asfortranarray(np.arange(9).reshape(3, 3),
                                   self.typeCode)
        self.assertEquals(second(matrix), 3)

    def testSecondElementObject(self):
        "Test luSplit function with a Fortran-array"
        print >>sys.stderr, self.typeStr, "... ",
        second = Fortran.__dict__[self.typeStr + "SecondElement"]
        matrix = np.asfortranarray([[0,1,2],[3,4,5],[6,7,8]], self.typeCode)
        self.assertEquals(second(matrix), 3)

######################################################################

class scharTestCase(FortranTestCase):
    def __init__(self, methodName="runTest"):
        FortranTestCase.__init__(self, methodName)
        self.typeStr  = "schar"
        self.typeCode = "b"

######################################################################

class ucharTestCase(FortranTestCase):
    def __init__(self, methodName="runTest"):
        FortranTestCase.__init__(self, methodName)
        self.typeStr  = "uchar"
        self.typeCode = "B"

######################################################################

class shortTestCase(FortranTestCase):
    def __init__(self, methodName="runTest"):
        FortranTestCase.__init__(self, methodName)
        self.typeStr  = "short"
        self.typeCode = "h"

######################################################################

class ushortTestCase(FortranTestCase):
    def __init__(self, methodName="runTest"):
        FortranTestCase.__init__(self, methodName)
        self.typeStr  = "ushort"
        self.typeCode = "H"

######################################################################

class intTestCase(FortranTestCase):
    def __init__(self, methodName="runTest"):
        FortranTestCase.__init__(self, methodName)
        self.typeStr  = "int"
        self.typeCode = "i"

######################################################################

class uintTestCase(FortranTestCase):
    def __init__(self, methodName="runTest"):
        FortranTestCase.__init__(self, methodName)
        self.typeStr  = "uint"
        self.typeCode = "I"

######################################################################

class longTestCase(FortranTestCase):
    def __init__(self, methodName="runTest"):
        FortranTestCase.__init__(self, methodName)
        self.typeStr  = "long"
        self.typeCode = "l"

######################################################################

class ulongTestCase(FortranTestCase):
    def __init__(self, methodName="runTest"):
        FortranTestCase.__init__(self, methodName)
        self.typeStr  = "ulong"
        self.typeCode = "L"

######################################################################

class longLongTestCase(FortranTestCase):
    def __init__(self, methodName="runTest"):
        FortranTestCase.__init__(self, methodName)
        self.typeStr  = "longLong"
        self.typeCode = "q"

######################################################################

class ulongLongTestCase(FortranTestCase):
    def __init__(self, methodName="runTest"):
        FortranTestCase.__init__(self, methodName)
        self.typeStr  = "ulongLong"
        self.typeCode = "Q"

######################################################################

class floatTestCase(FortranTestCase):
    def __init__(self, methodName="runTest"):
        FortranTestCase.__init__(self, methodName)
        self.typeStr  = "float"
        self.typeCode = "f"

######################################################################

class doubleTestCase(FortranTestCase):
    def __init__(self, methodName="runTest"):
        FortranTestCase.__init__(self, methodName)
        self.typeStr  = "double"
        self.typeCode = "d"

######################################################################

if __name__ == "__main__":

    # Build the test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(    scharTestCase))
    suite.addTest(unittest.makeSuite(    ucharTestCase))
    suite.addTest(unittest.makeSuite(    shortTestCase))
    suite.addTest(unittest.makeSuite(   ushortTestCase))
    suite.addTest(unittest.makeSuite(      intTestCase))
    suite.addTest(unittest.makeSuite(     uintTestCase))
    suite.addTest(unittest.makeSuite(     longTestCase))
    suite.addTest(unittest.makeSuite(    ulongTestCase))
    suite.addTest(unittest.makeSuite( longLongTestCase))
    suite.addTest(unittest.makeSuite(ulongLongTestCase))
    suite.addTest(unittest.makeSuite(    floatTestCase))
    suite.addTest(unittest.makeSuite(   doubleTestCase))

    # Execute the test suite
    print "Testing 2D Functions of Module Matrix"
    print "NumPy version", np.__version__
    print
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(len(result.errors) + len(result.failures))


"""
A buffered iterator for big arrays.

This module solves the problem of iterating over a big file-based array
without having to read it into memory. The ``Arrayterator`` class wraps
an array object, and when iterated it will return subarrays with at most
``buf_size`` elements.

The algorithm works by first finding a "running dimension", along which
the blocks will be extracted. Given an array of dimensions (d1, d2, ...,
dn), eg, if ``buf_size`` is smaller than ``d1`` the first dimension will
be used. If, on the other hand,

    d1 < buf_size < d1*d2

the second dimension will be used, and so on. Blocks are extracted along
this dimension, and when the last block is returned the process continues
