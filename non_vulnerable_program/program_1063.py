from weave_test_utils import *
restore_path()

build_dir = empty_temp_dir()
print 'building extensions here:', build_dir    

class test_ext_module(ScipyTestCase):
    #should really do some testing of where modules end up
    def check_simple(self,level=5):
        """ Simplest possible module """
        mod = ext_tools.ext_module('simple_ext_module')
        mod.compile(location = build_dir)
        import simple_ext_module
    def check_multi_functions(self,level=5):
        mod = ext_tools.ext_module('module_multi_function')
        var_specs = []
        code = ""
        test = ext_tools.ext_function_from_specs('test',code,var_specs)
        mod.add_function(test)
        test2 = ext_tools.ext_function_from_specs('test2',code,var_specs)
        mod.add_function(test2)
        mod.compile(location = build_dir)
        import module_multi_function
        module_multi_function.test()
        module_multi_function.test2()
    def check_with_include(self,level=5):
        # decalaring variables
        a = 2.;
    
        # declare module
        mod = ext_tools.ext_module('ext_module_with_include')
        mod.customize.add_header('<iostream>')
    
        # function 2 --> a little more complex expression
        var_specs = ext_tools.assign_variable_types(['a'],locals(),globals())
        code = """
               std::cout << std::endl;
               std::cout << "test printing a value:" << a << std::endl;
               """
        test = ext_tools.ext_function_from_specs('test',code,var_specs)
        mod.add_function(test)
        # build module
        mod.compile(location = build_dir)
        import ext_module_with_include
        ext_module_with_include.test(a)

    def check_string_and_int(self,level=5):        
        # decalaring variables
        a = 2;b = 'string'    
        # declare module
        mod = ext_tools.ext_module('ext_string_and_int')
        code = """
               a=b.length();
               return_val = PyInt_FromLong(a);
               """
        test = ext_tools.ext_function('test',code,['a','b'])
        mod.add_function(test)
        mod.compile(location = build_dir)
