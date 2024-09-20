import ext_tools

def build_fibonacci():
    """ Builds an extension module with fibonacci calculators.
    """
    mod = ext_tools.ext_module('fibonacci_ext')
    a = 1 # this is effectively a type declaration
    
    # recursive fibonacci in C 
    fib_code = """
                   int fib1(int a)
                   {                   
                       if(a <= 2)
                           return 1;
                       else
                           return fib1(a-2) + fib1(a-1);  
                   }                         
               """
    ext_code = """
                   int val = fib1(a);
                   return_val = Py::new_reference_to(Py::Int(val));
               """    
    fib = ext_tools.ext_function('c_fib1',ext_code,['a'])
    fib.customize.add_support_code(fib_code)
    mod.add_function(fib)

    # looping fibonacci in C
    fib_code = """
                    int fib2( int a )
                    {
                        int last, next_to_last, result;
            
                        if( a <= 2 )
                            return 1;            
                        last = next_to_last = 1;
                        for(int i = 2; i < a; i++ )
                        {
                            result = last + next_to_last;
                            next_to_last = last;
                            last = result;
                        }
            
                        return result;
                    }    
               """
    ext_code = """
                   int val = fib2(a);
                   return_val = Py::new_reference_to(Py::Int(val));
               """    
    fib = ext_tools.ext_function('c_fib2',ext_code,['a'])
    fib.customize.add_support_code(fib_code)
    mod.add_function(fib)       
    mod.compile()

try:
