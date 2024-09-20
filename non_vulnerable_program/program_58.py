import inline_tools
import scalar_spec
from blitz_tools import blitz_type_factories

def _cast_copy_transpose(type,a_2d):
    assert(len(shape(a_2d)) == 2)
    new_array = zeros(shape(a_2d),type)
    #trans_a_2d = transpose(a_2d)
    numeric_type = scalar_spec.numeric_to_blitz_type_mapping[type]
    code = """
           for(int i = 0; i < _Na_2d[0]; i++)
               for(int j = 0; j < _Na_2d[1]; j++)
                   new_array(i,j) = (%s) a_2d(j,i);
           """ % numeric_type
    inline_tools.inline(code,['new_array','a_2d'],
                        type_factories = blitz_type_factories,
                        compiler='gcc',
                        verbose = 1)
    return new_array

def _inplace_transpose(a_2d):
    assert(len(shape(a_2d)) == 2)
    numeric_type = scalar_spec.numeric_to_blitz_type_mapping[a_2d.typecode()]
    code = """
           %s temp;
           for(int i = 0; i < _Na_2d[0]; i++)
               for(int j = 0; j < _Na_2d[1]; j++)
               {
                   temp = a_2d(i,j);
                   a_2d(i,j) = a_2d(j,i);
                   a_2d(j,i) = temp;
               }    
           """ % numeric_type
    inline_tools.inline(code,['a_2d'],
                        type_factories = blitz_type_factories,
                        compiler='gcc',
                        verbose = 1)
    return a_2d

def cast_copy_transpose(type,*arrays):
    results = []
    for a in arrays:
        results.append(_cast_copy_transpose(type,a))
    if len(results) == 1:
        return results[0]
    else:
        return results

def inplace_cast_copy_transpose(*arrays):
    results = []
    for a in arrays:
        results.append(_inplace_transpose(a))
    if len(results) == 1:
        return results[0]
    else:
        return results

def _castCopyAndTranspose(type, *arrays):
    cast_arrays = ()
    for a in arrays:
        if a.typecode() == type:
            cast_arrays = cast_arrays + (copy.copy(Numeric.transpose(a)),)
        else:
            cast_arrays = cast_arrays + (copy.copy(
                                       Numeric.transpose(a).astype(type)),)
    if len(cast_arrays) == 1:
            return cast_arrays[0]
    else:
        return cast_arrays

