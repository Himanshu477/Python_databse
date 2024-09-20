        import ext_return_tuple
        c,d = ext_return_tuple.test(a)
        assert(c==a and d == a+1)
           
class test_ext_function(ScipyTestCase):
    #should really do some testing of where modules end up
    def check_simple(self,level=5):
        """ Simplest possible function """
        mod = ext_tools.ext_module('simple_ext_function')
        var_specs = []
        code = ""
        test = ext_tools.ext_function_from_specs('test',code,var_specs)
        mod.add_function(test)
        mod.compile(location = build_dir)
        import simple_ext_function
        simple_ext_function.test()
      
class test_assign_variable_types(ScipyTestCase):            
    def check_assign_variable_types(self):
        try:
            from scipy_base.numerix import arange, Float32, Float64
        except:
            # skip this test if scipy_base.numerix not installed
            return
            
        import types
        a = arange(10,typecode = Float32)
        b = arange(5,typecode = Float64)
        c = 5
        arg_list = ['a','b','c']
        actual = ext_tools.assign_variable_types(arg_list,locals())        
        #desired = {'a':(Float32,1),'b':(Float32,1),'i':(Int32,0)}
        
        ad = array_converter()
        ad.name, ad.var_type, ad.dims = 'a', Float32, 1
        bd = array_converter()
        bd.name, bd.var_type, bd.dims = 'b', Float64, 1

        cd = c_spec.int_converter()
        cd.name, cd.var_type = 'c', types.IntType        
        desired = [ad,bd,cd]
        expr = ""
        print_assert_equal(expr,actual,desired)

if __name__ == "__main__":
    ScipyTest().run()



