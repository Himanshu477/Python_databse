import os, c_spec # yes, I import myself to find out my __file__ location.
local_dir,junk = os.path.split(os.path.abspath(c_spec.__file__))   
scxx_dir = os.path.join(local_dir,'scxx')

class scxx_converter(common_base_converter):
    def init_info(self):
        common_base_converter.init_info(self)
        self.headers = ['"scxx/PWOBase.h"','"scxx/PWOSequence.h"',
                        '"scxx/PWOCallable.h"','"scxx/PWOMapping.h"',
                        '"scxx/PWOSequence.h"','"scxx/PWOMSequence.h"',
                        '"scxx/PWONumber.h"','<iostream>']
        self.include_dirs = [local_dir,scxx_dir]
        self.sources = [os.path.join(scxx_dir,'PWOImp.cpp'),]

class list_converter(scxx_converter):
    def init_info(self):
        scxx_converter.init_info(self)
        self.type_name = 'list'
        self.check_func = 'PyList_Check'    
        self.c_type = 'PWOList'
        self.to_c_return = 'PWOList(py_obj)'
        self.matching_types = [ListType]
        # ref counting handled by PWOList
        self.use_ref_count = 0

class tuple_converter(scxx_converter):
    def init_info(self):
        scxx_converter.init_info(self)
        self.type_name = 'tuple'
        self.check_func = 'PyTuple_Check'    
        self.c_type = 'PWOTuple'
        self.to_c_return = 'PWOTuple(py_obj)'
        self.matching_types = [TupleType]
        # ref counting handled by PWOTuple
        self.use_ref_count = 0

class dict_converter(scxx_converter):
    def init_info(self):
        scxx_converter.init_info(self)
        self.support_code.append("#define PWODict PWOMapping")
        self.type_name = 'dict'
        self.check_func = 'PyDict_Check'    
        self.c_type = 'PWODict'
        self.to_c_return = 'PWODict(py_obj)'
        self.matching_types = [DictType]
        # ref counting handled by PWODict
        self.use_ref_count = 0

#----------------------------------------------------------------------------
# Callable Converter
#----------------------------------------------------------------------------
class callable_converter(scxx_converter):
    def init_info(self):
        scxx_converter.init_info(self)
        self.type_name = 'callable'
        self.check_func = 'PyCallable_Check'    
        # probably should test for callable classes here also.
        self.matching_types = [FunctionType,MethodType,type(len)]
        self.c_type = 'PWOCallable'
        self.to_c_return = 'PWOCallable(py_obj)'
        # ref counting handled by PWOCallable
        self.use_ref_count = 0

def test(level=10):
    from scipy_base.testing import module_test
    module_test(__name__,__file__,level=level)

def test_suite(level=1):
    from scipy_base.testing import module_test_suite
    return module_test_suite(__name__,__file__,level=level)

if __name__ == "__main__":
    x = list_converter().type_spec("x",1)
    print x.py_to_c_code()
    print
    print x.c_to_py_code()
    print
    print x.declaration_code(inline=1)
    print
    print x.cleanup_code()

