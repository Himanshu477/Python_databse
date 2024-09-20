import weave
from weave import converters
import Numeric

def create_array():
    """Creates a simple 3D Numeric array with unique values at each
    location in the matrix.

    """    
    rows, cols, depth = 2, 3, 4
    arr = Numeric.zeros((rows, cols, depth), 'i')
    count = 0
    for i in range(rows):
        for j in range(cols):
            for k in range(depth):
                arr[i,j,k] = count
                count += 1
    return arr


def pure_inline(arr):
    """Prints the given 3D array by accessing the raw Numeric data and
    without using blitz converters.

    Notice the following:
      1. '\\n' to escape generating a newline in the C++ code.
      2. rows, cols = Narr[0], Narr[1].
      3. Array access using arr[(i*cols + j)*depth + k].
      
    """
    
    code = """
    int rows = Narr[0];
    int cols = Narr[1];
    int depth = Narr[2];    
    for (int i=0; i < rows; i++)
    {
        for (int j=0; j < cols; j++)
        {
            printf("img[%3d][%3d]=", i, j);
            for (int k=0; k< depth; ++k)
            {
                printf(" %3d", arr[(i*cols + j)*depth + k]);
            }
            printf("\\n");
        }
    }
    """

    weave.inline(code, ['arr'])


def blitz_inline(arr):
    """Prints the given 3D array by using blitz converters which
    provides a Numeric-like syntax for accessing the Numeric data.

    Notice the following:
      1. '\\n' to escape generating a newline in the C++ code.
      2. rows, cols = Narr[0], Narr[1].
      3. Array access using arr(i, j, k).
      
    """
    
    code = """
    int rows = Narr[0];
    int cols = Narr[1];
    int depth = Narr[2];    
    for (int i=0; i < rows; i++)
    {
        for (int j=0; j < cols; j++)
        {
            printf("img[%3d][%3d]=", i, j);
            for (int k=0; k< depth; ++k)
            {
                printf(" %3d", arr(i, j, k));
            }
            printf("\\n");
        }
    }
    """

    weave.inline(code, ['arr'], type_converters=converters.blitz)


def main():
    arr = create_array()
    print "Numeric:"    
    print arr

    print "Pure Inline:"
    pure_inline(arr)
    
    print "Blitz Inline:"
    blitz_inline(arr)
    

if __name__ == '__main__':
    main()


"""Simple example to show how to use weave.inline on SWIG2 wrapped
objects.  SWIG2 refers to SWIG versions >= 1.3.

To run this example you must build the trivial SWIG2 extension called
swig2_ext.  To do this you need to do something like this::

 $ swig -c++ -python -I. -o swig2_ext_wrap.cxx swig2_ext.i

 $ g++ -Wall -O2 -I/usr/include/python2.3 -fPIC -I. -c \
   -o swig2_ext_wrap.os swig2_ext_wrap.cxx

 $ g++ -shared -o _swig2_ext.so swig2_ext_wrap.os \
   -L/usr/lib/python2.3/config

The files swig2_ext.i and swig2_ext.h are included in the same
directory that contains this file.

Note that weave's SWIG2 support works fine whether SWIG_COBJECT_TYPES
are used or not.

Author: Prabhu Ramachandran
Copyright (c) 2004, Prabhu Ramachandran
License: BSD Style.

"""

# Import our SWIG2 wrapped library
