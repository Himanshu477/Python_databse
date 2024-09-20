from standard_array_spec import standard_array_factories

def vq3(obs,code_book):
    """ Uses standard array conversion completely bi-passing blitz.
        THIS DOES NOT HANDLE STRIDED ARRAYS CORRECTLY
    """
    # make sure we're looking at arrays.
    obs = asarray(obs)
    code_book = asarray(code_book)
    # check for 2d arrays and compatible sizes.
    obs_sh = shape(obs)
    code_book_sh = shape(code_book)
    assert(len(obs_sh) == 2 and len(code_book_sh) == 2)   
    assert(obs_sh[1] == code_book_sh[1])   
    assert(obs.typecode() == code_book.typecode())   
    type = scalar_spec.numeric_to_blitz_type_mapping[obs.typecode()]
    code =  """
            #line 139 "vq.py"
            // Surely there is a better way to do this...
            PyArrayObject* py_code = (PyArrayObject*) PyArray_FromDims(1,&_Nobs[0],PyArray_LONG);            
             PyArrayObject* py_min_dist = (PyArrayObject*) PyArray_FromDims(1,&_Nobs[0],PyArray_FLOAT);
             
            int* code_data = (int*)(py_code->data);
            float* min_dist_data = (float*)(py_min_dist->data);
            %(type)s* this_obs = NULL;
            %(type)s* this_code = NULL; 
            int Nfeatures = _Nobs[1];
            float diff,dist;

            for(int i=0; i < _Nobs[0]; i++)
            {
                this_obs = &obs_data[i*Nfeatures];
                min_dist_data[i] = (float)10000000.; // big number
                for(int j=0; j < _Ncode_book[0]; j++)
                {
                    this_code = &code_book_data[j*Nfeatures];
                    dist = 0;
                    for(int k=0; k < Nfeatures; k++)
                    {
                        diff = this_obs[k] - this_code[k];
                        dist +=  diff*diff;
                    }
                    if (dist < min_dist_data[i])
                    {
                        code_data[i] = j;
                        min_dist_data[i] = dist;                           
                    }    
                }
                min_dist_data[i] = sqrt(min_dist_data[i]);
             }
             Py::Tuple results(2);
             results[0] = Py::Object((PyObject*)py_code);
             results[1] = Py::Object((PyObject*)py_min_dist);
             return Py::new_reference_to(results);             
            """ % locals()
    # this is an unpleasant way to specify type factories -- work on it.
    import ext_tools
    code, distortion = inline_tools.inline(code,['obs','code_book'])
    return code, distortion

