from scipy.base.umath import minimum, maximum
del sys.path[0]


class test_maximum(ScipyTestCase):
    def check_reduce_complex(self):
        assert_equal(maximum.reduce([1,2j]),1)
        assert_equal(maximum.reduce([1+3j,2j]),1+3j)

class test_minimum(ScipyTestCase):
    def check_reduce_complex(self):
        assert_equal(minimum.reduce([1,2j]),2j)

if __name__ == "__main__":
    ScipyTest().run()


""" Test functions for limits module.

    Currently empty -- not sure how to test these values
    and routines as they are machine dependent.  Suggestions?
"""

