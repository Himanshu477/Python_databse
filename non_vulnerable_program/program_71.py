import ext_tools

def build_increment_ext():
    """ Build a simple extension with functions that increment numbers.
        The extension will be built in the local directory.
    """        
    mod = ext_tools.ext_module('increment_ext')

    a = 1 # effectively a type declaration for 'a' in the 
          # following functions.

    ext_code = "return_val = Py::new_reference_to(Py::Int(a+1));"    
    func = ext_tools.ext_function('increment',ext_code,['a'])
    mod.add_function(func)
    
    ext_code = "return_val = Py::new_reference_to(Py::Int(a+2));"    
    func = ext_tools.ext_function('increment_by_2',ext_code,['a'])
    mod.add_function(func)
            
    mod.compile()

if __name__ == "__main__":
    try:
