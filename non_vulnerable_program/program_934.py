import_array();
PyImport_ImportModule("scipy");
""" 
    
class array_converter(common_base_converter):

    def init_info(self):
        common_base_converter.init_info(self)
        self.type_name = 'numpy'
        self.check_func = 'PyArray_Check'    
        self.c_type = 'PyArrayObject*'
        self.return_type = 'PyArrayObject*'
        self.to_c_return = '(PyArrayObject*) py_obj'
        self.matching_types = [ArrayType]
        self.headers = ['"scipy/arrayobject.h"',
                        '<complex>','<math.h>']
        self.support_code = [size_check_code, type_check_code]
        self.module_init_code = [numeric_init_code]    
               
    def get_var_type(self,value):
        return value.dtypechar
    
    def template_vars(self,inline=0):
        res = common_base_converter.template_vars(self,inline)    
        if hasattr(self,'var_type'):
            res['num_type'] = num_to_c_types[self.var_type]
            res['num_typecode'] = num_typecode[self.var_type]
        res['array_name'] = self.name + "_array"
        return res
         
    def declaration_code(self,templatize = 0,inline=0):
        code = '%(py_var)s = %(var_lookup)s;\n'   \
               '%(c_type)s %(array_name)s = %(var_convert)s;\n'  \
               'conversion_numpy_check_type(%(array_name)s,%(num_typecode)s,"%(name)s");\n' \
               'int* N%(name)s = %(array_name)s->dimensions;\n' \
               'int* S%(name)s = %(array_name)s->strides;\n' \
               'int D%(name)s = %(array_name)s->nd;\n' \
               '%(num_type)s* %(name)s = (%(num_type)s*) %(array_name)s->data;\n' 
        code = code % self.template_vars(inline=inline)
        return code


#!/usr/bin/env python
"""

C declarations, CPP macros, and C functions for f2py2e.
Only required declarations/macros/functions will be used.

Copyright 1999,2000 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@ioc.ee>          
Permission to use, modify, and distribute this software is given under the
terms of the LGPL.  See http://www.fsf.org

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2005/05/06 11:42:34 $
Pearu Peterson
"""

__version__ = "$Revision: 1.75 $"[10:-1]

