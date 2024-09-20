from numpy.testing import *
import numpy as np

class TestDocs(NumpyTestCase):
    def check_doctests(self): return self.rundocs()

if __name__ == "__main__":
    NumpyTest().run()


