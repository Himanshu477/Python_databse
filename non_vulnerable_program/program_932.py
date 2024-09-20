import os, c_spec # yes, I import myself to find out my __file__ location.
local_dir,junk = os.path.split(os.path.abspath(c_spec.__file__))   
scxx_dir = os.path.join(local_dir,'scxx')

class scxx_converter(common_base_converter):
    def init_info(self):
        common_base_converter.init_info(self)
        self.headers = ['"scxx/object.h"','"scxx/list.h"','"scxx/tuple.h"',
                        '"scxx/dict.h"','<iostream>']
        self.include_dirs = [local_dir,scxx_dir]
        self.sources = [os.path.join(scxx_dir,'weave_imp.cpp'),]

class list_converter(scxx_converter):
    def init_info(self):
        scxx_converter.init_info(self)
        self.type_name = 'list'
        self.check_func = 'PyList_Check'    
        self.c_type = 'py::list'
        self.return_type = 'py::list'
        self.to_c_return = 'py::list(py_obj)'
        self.matching_types = [ListType]
        # ref counting handled by py::list
        self.use_ref_count = 0

class tuple_converter(scxx_converter):
    def init_info(self):
        scxx_converter.init_info(self)
        self.type_name = 'tuple'
        self.check_func = 'PyTuple_Check'    
        self.c_type = 'py::tuple'
        self.return_type = 'py::tuple'
        self.to_c_return = 'py::tuple(py_obj)'
        self.matching_types = [TupleType]
        # ref counting handled by py::tuple
        self.use_ref_count = 0

class dict_converter(scxx_converter):
    def init_info(self):
        scxx_converter.init_info(self)
        self.type_name = 'dict'
        self.check_func = 'PyDict_Check'    
        self.c_type = 'py::dict'
        self.return_type = 'py::dict'
        self.to_c_return = 'py::dict(py_obj)'
        self.matching_types = [DictType]
        # ref counting handled by py::dict
        self.use_ref_count = 0

#----------------------------------------------------------------------------
# Instance Converter
#----------------------------------------------------------------------------
class instance_converter(scxx_converter):
    def init_info(self):
        scxx_converter.init_info(self)
        self.type_name = 'instance'
        self.check_func = 'PyInstance_Check'    
        self.c_type = 'py::object'
        self.return_type = 'py::object'
        self.to_c_return = 'py::object(py_obj)'
        self.matching_types = [InstanceType]
        # ref counting handled by py::object
        self.use_ref_count = 0

#----------------------------------------------------------------------------
# Catchall Converter
#
# catch all now handles callable objects
#----------------------------------------------------------------------------
class catchall_converter(scxx_converter):
    def init_info(self):
        scxx_converter.init_info(self)
        self.type_name = 'catchall'
        self.check_func = ''    
        self.c_type = 'py::object'
        self.return_type = 'py::object'
        self.to_c_return = 'py::object(py_obj)'
        # ref counting handled by py::object
        self.use_ref_count = 0
    def type_match(self,value):
        return 1

if __name__ == "__main__":
    x = list_converter().type_spec("x",1)
    print x.py_to_c_code()
    print
    print x.c_to_py_code()
    print
    print x.declaration_code(inline=1)
    print
    print x.cleanup_code()


