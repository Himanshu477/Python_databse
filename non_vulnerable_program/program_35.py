import os
import UserList

class base_info:
    _warnings =[]
    _headers = []
    _include_dirs = []
    _libraries = []
    _library_dirs = []
    _support_code = []
    _module_init_code = []
    _sources = []
    _define_macros = []
    _undefine_macros = []
    compiler = ''
    def set_compiler(self,compiler):
        self.check_compiler(compiler)
        self.compiler = compiler
    # it would probably be better to specify what the arguments are
    # to avoid confusion, but I don't think these classes will get
    # very complicated, and I don't really know the variety of things
    # that should be passed in at this point.
    def check_compiler(self,compiler):
        pass        
    def warnings(self):   
        return self._warnings
    def headers(self):   
        return self._headers
    def include_dirs(self):
        return self._include_dirs
    def libraries(self):
        return self._libraries
    def library_dirs(self):
        return self._library_dirs
    def support_code(self):
        return self._support_code
    def module_init_code(self):
        return self._module_init_code
    def sources(self):
        return self._sources
    def define_macros(self):
        return self._define_macros
    def undefine_macros(self):
        return self._undefine_macros

class custom_info(base_info):
    def __init__(self):
        self._warnings =[]
        self._headers = []
        self._include_dirs = []
        self._libraries = []
        self._library_dirs = []
        self._support_code = []
        self._module_init_code = []
        self._sources = []
        self._define_macros = []
        self._undefine_macros = []

    def add_warning(self,warning):
        self._warnings.append(warning)
    def add_header(self,header):
        self._headers.append(header)
    def add_include_dir(self,include_dir):
        self._include_dirs.append(include_dir)
    def add_library(self,library):
        self._libraries.append(library)
    def add_library_dir(self,library_dir):
        self._library_dirs.append(library_dir)
    def add_support_code(self,support_code):
        self._support_code.append(support_code)
    def add_module_init_code(self,module_init_code):
        self._module_init_code.append(module_init_code)
    def add_source(self,source):
        self._sources.append(source)
    def add_define_macro(self,define_macro):
        self._define_macros.append(define_macro)
    def add_undefine_macro(self,undefine_macro):
        self._undefine_macros.append(undefine_macro)    

class info_list(UserList.UserList):
    def get_unique_values(self,attribute):
        all_values = []        
        for info in self:
            vals = eval('info.'+attribute+'()')
            all_values.extend(vals)
        return unique_values(all_values)
    
    def define_macros(self):
        return self.get_unique_values('define_macros')
    def sources(self):
        return self.get_unique_values('sources')
    def warnings(self):
        return self.get_unique_values('warnings')
    def headers(self):
        return self.get_unique_values('headers')
    def include_dirs(self):
        return self.get_unique_values('include_dirs')
    def libraries(self):
        return self.get_unique_values('libraries')
    def library_dirs(self):
        return self.get_unique_values('library_dirs')
    def support_code(self):
        return self.get_unique_values('support_code')
    def module_init_code(self):
        return self.get_unique_values('module_init_code')

def unique_values(lst):
    all_values = []        
    for value in lst:
        if value not in all_values:
            all_values.append(value)
    return all_values



class base_specification:
    """
        Properties:
        headers --  list of strings that name the header files needed by this
                    object.
        include_dirs -- list of directories where the header files can be found.
        libraries   -- list of libraries needed to link to when compiling
                       extension.
        library_dirs -- list of directories to search for libraries.

        support_code -- list of strings.  Each string is a subroutine needed
                        by the type.  Functions that are used in the conversion
                        between Python and C++ files are examples of these.

        Methods:

        type_match(value) returns 1 if this class is used to represent type
                          specification for value.
        type_spec(name, value)  returns a new object (of this class) that is
                                used to produce C++ code for value.
        declaration_code()    returns C++ code fragment for type declaration and
                              conversion of python object to C++ object.
        cleanup_code()    returns C++ code fragment for cleaning up after the
                          variable after main C++ code fragment has executed.

    """
    _build_information = []
    compiler = ''   
                
    def set_compiler(self,compiler):
        self.compiler = compiler
    def type_match(self,value):
        raise NotImplementedError, "You must override method in derived class"
    def build_information(self):
        return self._build_information
    def type_spec(self,name,value): pass
    def declaration_code(self,templatize = 0):   return ""
    def local_dict_code(self): return ""
    def cleanup_code(self): return ""
    def retrieve_py_variable(self,inline=0):
        # this needs a little coordination in name choices with the
        # ext_inline_function class.
        if inline:
            vn = 'get_variable("%s",raw_locals,raw_globals)' % self.name
        else:
            vn = 'py_' + self.name   
        return vn
        
    def py_reference(self):
        return "&py_" + self.name
    def py_pointer(self):
        return "*py_" + self.name
    def py_variable(self):
        return "py_" + self.name
    def reference(self):
        return "&" + self.name
    def pointer(self):
        return "*" + self.name
    def variable(self):
        return self.name
    def variable_as_string(self):
        return '"' + self.name + '"'

