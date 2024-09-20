import inline_tools
from blitz_tools import blitz_type_factories
import scalar_spec

def vq(obs,code_book):
    # make sure we're looking at arrays.
    obs = asarray(obs)
    code_book = asarray(code_book)
    # check for 2d arrays and compatible sizes.
    obs_sh = shape(obs)
    code_book_sh = shape(code_book)
    assert(len(obs_sh) == 2 and len(code_book_sh) == 2)   
    assert(obs_sh[1] == code_book_sh[1])   
    type = scalar_spec.numeric_to_blitz_type_mapping[obs.typecode()]
    # band aid for now.
    ar_type = 'PyArray_FLOAT'
    code =  """
            #line 37 "vq.py"
            // Use tensor notation.            
            blitz::Array<%(type)s,2> dist_sq(_Ncode_book[0],_Nobs[0]);
             blitz::firstIndex i;    
            blitz::secondIndex j;   
            blitz::thirdIndex k;
            dist_sq = sum(pow2(obs(j,k) - code_book(i,k)),k);
            // Surely there is a better way to do this...
            PyArrayObject* py_code = (PyArrayObject*) PyArray_FromDims(1,&_Nobs[0],PyArray_LONG);
             blitz::Array<int,1> code((int*)(py_code->data),
                                     blitz::shape(_Nobs[0]), blitz::neverDeleteData);
             code = minIndex(dist_sq(j,i),j);
             
             PyArrayObject* py_min_dist = (PyArrayObject*) PyArray_FromDims(1,&_Nobs[0],PyArray_FLOAT);
             blitz::Array<float,1> min_dist((float*)(py_min_dist->data),
                                            blitz::shape(_Nobs[0]), blitz::neverDeleteData);
             min_dist = sqrt(min(dist_sq(j,i),j));
             Py::Tuple results(2);
             results[0] = Py::Object((PyObject*)py_code);
             results[1] = Py::Object((PyObject*)py_min_dist);
             return_val = Py::new_reference_to(results);             
            """ % locals()
    code, distortion = inline_tools.inline(code,['obs','code_book'],
                                           type_factories = blitz_type_factories,
                                           compiler = 'gcc',
                                           verbose = 1)
    return code, distortion

def vq2(obs,code_book):
    """ doesn't use blitz (except in conversion)
        ALSO DOES NOT HANDLE STRIDED ARRAYS CORRECTLY
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
    # band aid for now.
    ar_type = 'PyArray_FLOAT'
    code =  """
            #line 83 "vq.py"
            // THIS DOES NOT HANDLE STRIDED ARRAYS CORRECTLY
            // Surely there is a better way to do this...
            PyArrayObject* py_code = (PyArrayObject*) PyArray_FromDims(1,&_Nobs[0],PyArray_LONG);            
             PyArrayObject* py_min_dist = (PyArrayObject*) PyArray_FromDims(1,&_Nobs[0],PyArray_FLOAT);
             
            int* raw_code = (int*)(py_code->data);
            float* raw_min_dist = (float*)(py_min_dist->data);
            %(type)s* raw_obs = obs.data();
            %(type)s* raw_code_book = code_book.data(); 
            %(type)s* this_obs = NULL;
            %(type)s* this_code = NULL; 
            int Nfeatures = _Nobs[1];
            float diff,dist;
            for(int i=0; i < _Nobs[0]; i++)
            {
                this_obs = &raw_obs[i*Nfeatures];
                raw_min_dist[i] = (%(type)s)10000000.; // big number
                for(int j=0; j < _Ncode_book[0]; j++)
                {
                    this_code = &raw_code_book[j*Nfeatures];
                    dist = 0;
                    for(int k=0; k < Nfeatures; k++)
                    {
                        diff = this_obs[k] - this_code[k];
                        dist +=  diff*diff;
                    }
                    dist = dist;
                    if (dist < raw_min_dist[i])
                    {
                        raw_code[i] = j;
                        raw_min_dist[i] = dist;                           
                    }    
                }
                raw_min_dist[i] = sqrt(raw_min_dist[i]);
             }
             Py::Tuple results(2);
             results[0] = Py::Object((PyObject*)py_code);
             results[1] = Py::Object((PyObject*)py_min_dist);
             return_val = Py::new_reference_to(results);             
            """ % locals()
    code, distortion = inline_tools.inline(code,['obs','code_book'],
                                         type_factories = blitz_type_factories,
                                         compiler = 'gcc',
                                         verbose = 1)
    return code, distortion

