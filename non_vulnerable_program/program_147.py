        import ext_string_and_int
        c = ext_string_and_int.test(a,b)
        assert(c == len(b))
        
    def check_return_tuple(self):        
        # decalaring variables
        a = 2    
        # declare module
        mod = ext_tools.ext_module('ext_return_tuple')
        var_specs = ext_tools.assign_variable_types(['a'],locals())
        code = """
               int b;
               b = a + 1;
               Py::Tuple returned(2);
               returned[0] = Py::Int(a);
               returned[1] = Py::Int(b);
               return_val = Py::new_reference_to(returned);
               """
        test = ext_tools.ext_function('test',code,['a'])
        mod.add_function(test)
        mod.compile(location = build_dir)
