from numpy.fft import *
restore_path()

class test_fftshift(NumpyTestCase):
    def check_fft_n(self):
        self.failUnlessRaises(ValueError,fft,[1,2,3],0)

if __name__ == "__main__":
    NumpyTest().run()


""" Build swig, f2py, weave, sources.
"""

