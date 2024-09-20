import_array();
if (PyErr_Occurred()) {
  PyErr_SetString(PyExc_ImportError, "failed to load array module.");
  goto capi_error;
}
''', provides='import_array')

    )



