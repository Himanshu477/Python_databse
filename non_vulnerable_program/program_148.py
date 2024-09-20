        import ext_return_tuple
        c,d = ext_return_tuple.test(a)
        assert(c==a and d == a+1)
           
class test_ext_function(unittest.TestCase):
    #should really do some testing of where modules end up
    def check_simple(self):
        """ Simplest possible function """
        mod = ext_tools.ext_module('simple_ext_function')
        var_specs = []
        code = ""
        test = ext_tools.ext_function_from_specs('test',code,var_specs)
        mod.add_function(test)
        mod.compile(location = build_dir)
        import simple_ext_function
        simple_ext_function.test()
      
class test_assign_variable_types(unittest.TestCase):            
    def check_assign_variable_types(self):
        try:
            from Numeric import *
        except:
            # skip this test if Numeric not installed
            return
            
        import types
        a = arange(10,typecode = Float32)
        b = arange(5,typecode = Float64)
        c = 5
        arg_list = ['a','b','c']
        actual = ext_tools.assign_variable_types(arg_list,locals())        
        #desired = {'a':(Float32,1),'b':(Float32,1),'i':(Int32,0)}
        
        ad = array_specification()
        ad.name, ad.numeric_type, ad.dims = 'a', Float32, 1
        bd = array_specification()
        bd.name, bd.numeric_type, bd.dims = 'b', Float64, 1
        import scalar_spec
        cd = scalar_spec.int_specification()
        cd.name, cd.numeric_type = 'c', types.IntType        
        desired = [ad,bd,cd]
        expr = ""
        print_assert_equal(expr,actual,desired)


def test_suite():
    suites = []
    suites.append( unittest.makeSuite(test_assign_variable_types,'check_') )
    suites.append( unittest.makeSuite(test_ext_module,'check_'))
    suites.append( unittest.makeSuite(test_ext_function,'check_'))      
    total_suite = unittest.TestSuite(suites)
    return total_suite

def test():
    all_tests = test_suite()
    runner = unittest.TextTestRunner()
    runner.run(all_tests)
    return runner

if __name__ == "__main__":
    test()


