import numpy as np
from numpy.testing import *

class TestPrint(TestCase):

    def test_float_types(self) :
        """ Check formatting.

            This is only for the str function, and only for simple types.
            The precision of np.float and np.longdouble aren't the same as the
            python float precision.

        """
        for t in [np.float, np.double, np.longdouble] :
            for x in [0, 1,-1, 1e10, 1e20] :
                assert_equal(str(t(x)), str(float(x)))

    def test_complex_types(self) :
        """Check formatting.

            This is only for the str function, and only for simple types.
            The precision of np.float and np.longdouble aren't the same as the
            python float precision.

        """
        for t in [np.cfloat, np.cdouble, np.clongdouble] :
            for x in [0, 1,-1, 1e10, 1e20] :
                assert_equal(str(t(x)), str(complex(x)))
                assert_equal(str(t(x*1j)), str(complex(x*1j)))
                assert_equal(str(t(x + x*1j)), str(complex(x + x*1j)))




if __name__ == "__main__":
    run_module_suite()


"""

=================
Glossary
=================

Place-holder for a glossary.

"""


"""

======
Jargon
======

Placeholder for computer science, engineering and other jargon.

"""


# These classes implement a doctest runner plugin for nose.
# Because this module imports nose directly, it should not
# be used except by nosetester.py to avoid a general NumPy
# dependency on nose.

