from numpy.testing import *
import numpy as np

class TestDocs(NumpyTestCase):
    def check_doctests(self): return self.rundocs()

    def check_roots(self):
        assert_array_equal(np.roots([1,0,0]), [0,0])

    def check_str_leading_zeros(self):
        p = np.poly1d([4,3,2,1])
        p[3] = 0
        assert_equal(str(p),
                     "   2\n"
                     "3 x + 2 x + 1")

        p = np.poly1d([1,2])
        p[0] = 0
        p[1] = 0
        assert_equal(str(p), " \n0")

if __name__ == "__main__":
    NumpyTest().run()



__all__ = ['savetxt', 'loadtxt',
           'load', 'loads',
           'save', 'savez',
           'packbits', 'unpackbits',
           'DataSource']

