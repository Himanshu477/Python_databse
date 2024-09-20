import_array(void) 
{ 
  PyObject *numpy = PyImport_ImportModule("scipy.base.multiarray");
  PyObject *c_api = NULL;
  if (numpy == NULL) return -1;
  c_api = PyObject_GetAttrString(numpy, "_ARRAY_API");
  if (c_api == NULL) {Py_DECREF(numpy); return -1;}
  if (PyCObject_Check(c_api)) { 
      PyArray_API = (void **)PyCObject_AsVoidPtr(c_api); 
  }
  Py_DECREF(c_api);
  Py_DECREF(numpy);
  if (PyArray_API == NULL) return -1;
  return 0;
}

#endif

""" % ('\n'.join(module_list), 
       '\n'.join(extension_list))

# Write to header
fid = open('__multiarray_api.h','w')
fid.write(outstr)
fid.close()


outstr = r"""
/* These pointers will be stored in the C-object for use in other
    extension modules
*/

void *PyArray_API[] = {
        (void *) &PyArray_Type,
        (void *) &PyArrayIter_Type,
        (void *) &PyArrayMapIter_Type,
%s
};
""" % '\n'.join(init_list)

# Write to c-code
fid = open('__multiarray_api.c','w')
fid.write(outstr)
fid.close()






