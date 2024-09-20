import inline_tools
from bisect import bisect

def c_int_search(seq,t,chk=1):
    # do partial type checking in Python.
    # checking that list items are ints should happen in py_to_scalar<int>
    if chk:
        assert(type(t) == type(1))
        assert(type(seq) == type([]))
    code = """     
           #line 29 "binary_search.py"
           int val, m, min = 0; 
           int max = seq.length()- 1;
           PyObject *py_val;
           for(;;) 
           { 
               if (max < min )
               {
                   return_val = Py::new_reference_to(Py::Int(-1));
                   break;
               }
               m = (min + max) / 2;
               val = py_to_int(PyList_GetItem(seq.ptr(),m),"val");
               if (val < t)     
                   min = m + 1;
               else if (val > t)    
                   max = m - 1;
               else
               {
                   return_val = Py::new_reference_to(Py::Int(m));
                   break;
               }
           }      
           """    
    #return inline_tools.inline(code,['seq','t'],compiler='msvc')
    return inline_tools.inline(code,['seq','t'],verbose = 2)

try:
