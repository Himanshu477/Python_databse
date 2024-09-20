import inline_tools

def c_sort(adict):
    assert(type(adict) == type({}))
    code = """
           #line 21 "dict_sort.py"     
           Py::List keys = adict.keys();
           Py::List items(keys.length());
           keys.sort(); // surely this isn't any slower than raw API calls
           PyObject* item = NULL;
           for(int i = 0; i < keys.length();i++)
           {
              item = PyList_GET_ITEM(keys.ptr(),i);
              item = PyDict_GetItem(adict.ptr(),item);
              Py_XINCREF(item);
              PyList_SetItem(items.ptr(),i,item);              
           }           
           return_val = Py::new_reference_to(items);
           """   
    return inline_tools.inline(code,['adict'],verbose=1)


# (IMHO) the simplest approach:
def sortedDictValues1(adict):
    items = adict.items()
    items.sort()
    return [value for key, value in items]

# an alternative implementation, which
# happens to run a bit faster for large
# dictionaries on my machine:
def sortedDictValues2(adict):
    keys = adict.keys()
    keys.sort()
    return [adict[key] for key in keys]

# a further slight speed-up on my box
# is to map a bound-method:
def sortedDictValues3(adict):
    keys = adict.keys()
    keys.sort()
    return map(adict.get, keys)

