import UserList
import base_info

class arg_spec_list(UserList.UserList):    
    def build_information(self): 
        all_info = base_info.info_list()
        for i in self:
            all_info.extend(i.build_information())
        return all_info
        
    def py_references(self): return map(lambda x: x.py_reference(),self)
    def py_pointers(self): return map(lambda x: x.py_pointer(),self)
    def py_variables(self): return map(lambda x: x.py_variable(),self)

    def references(self): return map(lambda x: x.py_reference(),self)
    def pointers(self): return map(lambda x: x.pointer(),self)    
    def variables(self): return map(lambda x: x.variable(),self)
    
    def variable_as_strings(self): 
        return map(lambda x: x.variable_as_string(),self)

    

"""
    build_info holds classes that define the information
    needed for building C++ extension modules for Python that
    handle different data types.  The information includes
    such as include files, libraries, and even code snippets.
       
    array_info -- for building functions that use Python
                  Numeric arrays.
"""

