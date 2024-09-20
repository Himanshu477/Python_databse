import doctest
def test_suite(level=1):
    return doctest.DocTestSuite()

if __name__ == "__main__":
    ScipyTest().run()


## Automatically adapted for numpy Sep 19, 2005 by convertcode.py

__all__ = ['iscomplexobj','isrealobj','imag','iscomplex',
           'isscalar',
           'isreal','nan_to_num','real','real_if_close',
           'typename','asfarray','mintypecode','asscalar',
           'common_type']

