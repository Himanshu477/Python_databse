import standard_array_info
import os, blitz_info
local_dir,junk = os.path.split(os.path.abspath(blitz_info.__file__))   
blitz_dir = os.path.join(local_dir,'blitz-20001213')

class array_info(base_info.base_info):
    _include_dirs = [blitz_dir]
    _headers = ['"blitz/array.h"','"Numeric/arrayobject.h"','<complex>','<math.h>']
    
    _support_code = [standard_array_info.array_convert_code,
                     standard_array_info.type_check_code,
                     standard_array_info.size_check_code,
                     blitz_support_code]
    _module_init_code = [standard_array_info.numeric_init_code]    
    
    # throw error if trying to use msvc compiler
    
    def check_compiler(self,compiler):        
        msvc_msg = 'Unfortunately, the blitz arrays used to support numeric' \
                   ' arrays will not compile with MSVC.' \
                   '  Please try using mingw32 (www.mingw.org).'
        if compiler == 'msvc':
            return ValueError, self.msvc_msg        

