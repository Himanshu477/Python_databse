from weave import standard_array_spec
restore_path()

def remove_whitespace(in_str):
    import string
    out = string.replace(in_str," ","")
    out = string.replace(out,"\t","")
    out = string.replace(out,"\n","")
    return out
    
def print_assert_equal(test_string,actual,desired):
    """this should probably be in scipy_test.testing
    """
    import pprint
    try:
        assert(actual == desired)
    except AssertionError:
        import cStringIO
        msg = cStringIO.StringIO()
        msg.write(test_string)
        msg.write(' failed\nACTUAL: \n')
        pprint.pprint(actual,msg)
        msg.write('DESIRED: \n')
        pprint.pprint(desired,msg)
        raise AssertionError, msg.getvalue()

class test_array_converter(ScipyTestCase):    
    def check_type_match_string(self):
        s = standard_array_spec.array_converter()
        assert( not s.type_match('string') )
    def check_type_match_int(self):
        s = standard_array_spec.array_converter()        
        assert(not s.type_match(5))
    def check_type_match_array(self):
        s = standard_array_spec.array_converter()        
        assert(s.type_match(arange(4)))

if __name__ == "__main__":
    ScipyTest().run()



"""
check_var_in -- tests whether a variable is passed in correctly
                and also if the passed in variable can be reassigned
check_var_local -- tests wheter a variable is passed in , modified,
                   and returned correctly in the local_dict dictionary
                   argument
check_return -- test whether a variable is passed in, modified, and
                then returned as a function return value correctly
"""

