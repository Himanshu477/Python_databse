import weave
from weave import swig2_spec, converters

# SWIG2 support is not enabled by default.  We do this by adding the
# swig2 converter to the default list of converters.
converters.default.insert(0, swig2_spec.swig2_converter())

def test():
    """Instantiate the SWIG wrapped object and then call its method
    from C++ using weave.inline
    
    """
    a = swig2_ext.A()
    b = swig2_ext.foo()  # This will be an APtr instance.
    b.thisown = 1 # Prevent memory leaks.
    code = """a->f();
              b->f();
              """
    weave.inline(code, ['a', 'b'], include_dirs=['.'], 
                 headers=['"swig2_ext.h"'], verbose=1)

    
if __name__ == "__main__":
    test()


""" A simple example to show how to use weave with VTK.  This lets one
create VTK objects using the standard VTK-Python API (via 'import
vtk') and then accelerate any of the computations by inlining C++ code
inside Python.

Please note the use of the `inc_dirs` and the `lib_dirs` variables in
the call to weave.inline.  Point these to where your VTK headers are
and where the shared libraries are.

For every VTK object encountered the corresponding VTK header is
automatically added to the C++ code generated.  If you need to add
other headers specified like so::

 headers=['"vtkHeader1.h"', '"vtkHeader2.h"']

in the keyword arguments to weave.inline.  Similarly, by default,
vtkCommon is linked into the generated module.  If you need to link to
any of the other vtk libraries add something like so::

 libraries=['vtkHybrid', 'vtkFiltering']

in the keyword arguments to weave.inline.  For example::

 weave.inline(code, ['arr', 'v_arr'],
              include_dirs = ['/usr/local/include/vtk'],
              library_dirs = ['/usr/local/lib/vtk'],
              headers=['"vtkHeader1.h"', '"vtkHeader2.h"'],
              libraries=['vtkHybrid', 'vtkFiltering'])


This module has been tested to work with VTK-4.2 and VTK-4.4 under
Linux.  YMMV on other platforms.


Author: Prabhu Ramachandran
Copyright (c) 2004, Prabhu Ramachandran
License: BSD Style.

"""

