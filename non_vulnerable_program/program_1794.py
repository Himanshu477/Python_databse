from numpy import *

class test_m(NumpyTestCase):

    def check_foo_simple(self, level=1):
        foo = m.foo
        r = foo(2)
        assert isinstance(r,int32),`type(r)`
        assert_equal(r,3)

    def check_foo2_simple(self, level=1):
        foo2 = m.foo2
        r = foo2(2)
        assert isinstance(r,int32),`type(r)`
        assert_equal(r,4)

if __name__ == "__main__":
    NumpyTest().run()


#~ import sys
#~ if sys.version_info[:2] >= (2, 5):
    #~ exec """
