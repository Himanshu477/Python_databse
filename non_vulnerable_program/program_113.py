import_array();
PyImport_ImportModule("Numeric");
"""

class array_info(base_info.base_info):
    _headers = ['"Numeric/arrayobject.h"','<complex>','<math.h>']
    _support_code = [array_convert_code,size_check_code, type_check_code]
    _module_init_code = [numeric_init_code]    

