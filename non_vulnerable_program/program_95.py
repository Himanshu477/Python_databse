    import wx_spec
    default_type_factories.append(wx_spec.wx_specification())
except: 
    pass    

class ext_function_from_specs:
    def __init__(self,name,code_block,arg_specs):
        self.name = name
        self.arg_specs = base_spec.arg_spec_list(arg_specs)
        self.code_block = code_block
        self.compiler = ''
        self.customize = base_info.custom_info()
        
    def header_code(self):
        pass

    def function_declaration_code(self):
       code  = 'static PyObject* %s(PyObject*self, PyObject* args,' \
               ' PyObject* kywds)\n{\n'
       return code % self.name

    def template_declaration_code(self):
        code = 'template<class T>\n' \
               'static PyObject* %s(PyObject*self, PyObject* args,' \
               ' PyObject* kywds)\n{\n'
        return code % self.name

    #def cpp_function_declaration_code(self):
    #    pass
    #def cpp_function_call_code(self):
    #s    pass
        
    def parse_tuple_code(self):
        """ Create code block for PyArg_ParseTuple.  Variable declarations
            for all PyObjects are done also.
            
            This code got a lot uglier when I added local_dict...
        """
        join = string.join

        declare_return = 'PyObject *return_val = NULL;\n' \
                         'int exception_occured = 0;\n' \
                         'PyObject *py_local_dict = NULL;\n'
        arg_string_list = self.arg_specs.variable_as_strings() + ['"local_dict"']
        arg_strings = join(arg_string_list,',')
        if arg_strings: arg_strings += ','
        declare_kwlist = 'static char *kwlist[] = {%s NULL};\n' % arg_strings

        py_objects = join(self.arg_specs.py_pointers(),', ')
        if py_objects:
            declare_py_objects = 'PyObject ' + py_objects +';\n'
        else:
            declare_py_objects = ''
            
        py_vars = join(self.arg_specs.py_variables(),' = ')
        if py_vars:
            init_values = py_vars + ' = NULL;\n\n'
        else:
            init_values = ''    

        #Each variable is in charge of its own cleanup now.
        #cnt = len(arg_list)
        #declare_cleanup = "blitz::TinyVector<PyObject*,%d> clean_up(0);\n" % cnt

        ref_string = join(self.arg_specs.py_references(),', ')
        if ref_string:
            ref_string += ', &py_local_dict'
        else:
            ref_string = '&py_local_dict'
            
        format = "O"* len(self.arg_specs) + "|O" + ':' + self.name
        parse_tuple =  'if(!PyArg_ParseTupleAndKeywords(args,' \
                             'kywds,"%s",kwlist,%s))\n' % (format,ref_string)
        parse_tuple += '   return NULL;\n'

        return   declare_return + declare_kwlist + declare_py_objects  \
               + init_values + parse_tuple

    def arg_declaration_code(self):
        arg_strings = []
        for arg in self.arg_specs:
            arg_strings.append(arg.declaration_code())
        code = string.join(arg_strings,"")
        return code

    def arg_cleanup_code(self):
        arg_strings = []
        for arg in self.arg_specs:
            arg_strings.append(arg.cleanup_code())
        code = string.join(arg_strings,"")
        return code

    def arg_local_dict_code(self):
        arg_strings = []
        for arg in self.arg_specs:
            arg_strings.append(arg.local_dict_code())
        code = string.join(arg_strings,"")
        return code
        
    def function_code(self):
        decl_code = indent(self.arg_declaration_code(),4)
        cleanup_code = indent(self.arg_cleanup_code(),4)
        function_code = indent(self.code_block,4)
        local_dict_code = indent(self.arg_local_dict_code(),4)

        dict_code = "if(py_local_dict)                                  \n"   \
                    "{                                                  \n"   \
                    "    Py::Dict local_dict = Py::Dict(py_local_dict); \n" + \
                         local_dict_code                                    + \
                    "}                                                  \n"

        try_code =    "try                              \n"   \
                      "{                                \n" + \
                           decl_code                        + \
                      "    /*<function call here>*/     \n" + \
                           function_code                    + \
                           indent(dict_code,4)              + \
                      "\n}                                \n"
        catch_code =  "catch( Py::Exception& e)           \n"   \
                      "{                                \n" + \
                      "    return_val =  Py::Null();    \n"   \
                      "    exception_occured = 1;       \n"   \
                      "}                                \n"

        return_code = "    /*cleanup code*/                     \n" + \
                           cleanup_code                             + \
                      "    if(!return_val && !exception_occured)\n"   \
                      "    {\n                                  \n"   \
                      "        Py_INCREF(Py_None);              \n"   \
                      "        return_val = Py_None;            \n"   \
                      "    }\n                                  \n"   \
                      "    return return_val;           \n"           \
                      "}                                \n"

        all_code = self.function_declaration_code()         + \
                       indent(self.parse_tuple_code(),4)    + \
                       indent(try_code,4)                   + \
                       indent(catch_code,4)                 + \
                       return_code

        return all_code

    def python_function_definition_code(self):
        args = (self.name, self.name)
        function_decls = '{"%s",(PyCFunction)%s , METH_VARARGS|' \
                          'METH_KEYWORDS},\n' % args
        return function_decls

    def set_compiler(self,compiler):
        self.compiler = compiler
        for arg in self.arg_specs:
            arg.set_compiler(compiler)


class ext_function(ext_function_from_specs):
    def __init__(self,name,code_block, args, local_dict=None, global_dict=None,
                 auto_downcast=1, type_factories=None):
                    
        call_frame = sys._getframe().f_back
        if local_dict is None:
            local_dict = call_frame.f_locals
        if global_dict is None:
            global_dict = call_frame.f_globals
        if type_factories is None:
            type_factories = default_type_factories
        arg_specs = assign_variable_types(args,local_dict, global_dict,
                                          auto_downcast, type_factories)
        ext_function_from_specs.__init__(self,name,code_block,arg_specs)
        
            
