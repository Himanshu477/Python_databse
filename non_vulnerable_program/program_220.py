    from Numeric import *
    from fastumath import *
    
    def assert_array_equal(x,y,err_msg=''):
        msg = '\nArrays are not equal'
        try:
            assert alltrue(equal(shape(x),shape(y))),\
                   msg + ' (shapes mismatch):\n\t' + err_msg
            reduced = equal(x,y)
            assert alltrue(ravel(reduced)),\
                   msg + ':\n\t' + err_msg
        except ValueError:
            print shape(x),shape(y)
            raise ValueError, 'arrays are not equal'
    
    def assert_array_almost_equal(x,y,decimal=6,err_msg=''):
        msg = '\nArrays are not almost equal'
        try:
            assert alltrue(equal(shape(x),shape(y))),\
                   msg + ' (shapes mismatch):\n\t' + err_msg
            reduced = equal(around(abs(x-y),decimal),0)
            assert alltrue(ravel(reduced)),\
                   msg + ':\n\t' + err_msg
        except ValueError:
            print sys.exc_value
            print shape(x),shape(y)
            print x, y
            raise ValueError, 'arrays are not almost equal'
except:
    pass # Numeric not installed
    
