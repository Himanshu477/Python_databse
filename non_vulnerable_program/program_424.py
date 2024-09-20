import sys
import time


# Please change these to suit your needs.  If not, this example will
# not compile.
inc_dirs = ['/usr/local/include/vtk', '/usr/include/vtk']
lib_dirs = ['/usr/local/lib/vtk', '/usr/lib/vtk']


def simple_test():
    """A simple example of how you can access the methods of a VTK
    object created from Python in C++ using weave.inline.

    """
    
    a = vtk.vtkStructuredPoints()
    a.SetOrigin(1.0, 1.0, 1.0)
    print "sys.getrefcount(a) = ", sys.getrefcount(a)

    code=r"""
    printf("a->ClassName() == %s\n", a->GetClassName());
    printf("a->GetReferenceCount() == %d\n", a->GetReferenceCount());
    double *origin = a->GetOrigin();
    printf("Origin = %f, %f, %f\n", origin[0], origin[1], origin[2]);
    """
    weave.inline(code, ['a'], include_dirs=inc_dirs, library_dirs=lib_dirs)

    print "sys.getrefcount(a) = ", sys.getrefcount(a)
    

def array_test():
    """Tests if a large Numeric array can be copied into a
    vtkFloatArray rapidly by using weave.inline.

    """

    # Create a large Numeric array.
    arr = Numeric.arange(0, 10, 0.0001, 'f')
    print "Number of elements in array = ", arr.shape[0]

    # Copy it into a vtkFloatArray and time the process.
    v_arr = vtk.vtkFloatArray()
    ts = time.clock()
    for i in range(arr.shape[0]):
        v_arr.InsertNextValue(arr[i])
    print "Time taken to do it in pure Python =", time.clock() - ts    

    # Now do the same thing using weave.inline
    v_arr = vtk.vtkFloatArray()
    code = """
    int size = Narr[0];
    for (int i=0; i<size; ++i)
        v_arr->InsertNextValue(arr[i]);
    """
    ts = time.clock()
    # Note the use of the include_dirs and library_dirs.
    weave.inline(code, ['arr', 'v_arr'], include_dirs=inc_dirs,
                 library_dirs=lib_dirs)    
    print "Time taken to do it using Weave =", time.clock() - ts

    # Test the data to make certain that we have done it right.
    print "Checking data."
    for i in range(v_arr.GetNumberOfTuples()):
        val = (v_arr.GetValue(i) -arr[i] )
        assert (val < 1e-6), "i = %d, val= %f"%(i, val)
    print "OK."


if __name__ == "__main__":    
    simple_test()
    array_test()


"""Imports from numarray for numerix, the numarray/Numeric interchangeability
module.  These array functions are used when numarray is chosen.
"""
