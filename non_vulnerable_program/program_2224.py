imported from Python. It will contain a method for each subroutin in
add (zadd, cadd, dadd, sadd). The docstring of each method contains
information about how the module method may be called:

    >>> import add
    >>> print add.zadd.__doc__
    zadd - Function signature: 
      zadd(a,b,c,n)
    Required arguments: 
      a : input rank-1 array('D') with bounds (*)
      b : input rank-1 array('D') with bounds (*)
      c : input rank-1 array('D') with bounds (*)
      n : input int   


Improving the basic interface
-----------------------------

The default interface is a very literal translation of the fortran
code into Python. The Fortran array arguments must now be NumPy arrays
and the integer argument should be an integer. The interface will
attempt to convert all arguments to their required types (and shapes)
and issue an error if unsuccessful. However, because it knows nothing
about the semantics of the arguments (such that C is an output and n
should really match the array sizes), it is possible to abuse this
function in ways that can cause Python to crash. For example: 

    >>> add.zadd([1,2,3],[1,2],[3,4],1000)

will cause a program crash on most systems. Under the covers, the
lists are being converted to proper arrays but then the underlying add
loop is told to cycle way beyond the borders of the allocated memory. 

In order to improve the interface, directives should be provided. This
is accomplished by constructing an interface definition file. It is
usually best to start from the interface file that f2py can produce
(where it gets its default behavior from). To get f2py to generate the
interface file use the -h option::

    f2py -h add.pyf -m add add.f

This command leaves the file add.pyf in the current directory. The
section of this file corresponding to zadd is:

.. code-block:: none

    subroutine zadd(a,b,c,n) ! in :add:add.f 
       double complex dimension(*) :: a 
       double complex dimension(*) :: b 
       double complex dimension(*) :: c 
       integer :: n 
    end subroutine zadd

By placing intent directives and checking code, the interface can be
cleaned up quite a bit until the Python module method is both easier
to use and more robust.

.. code-block:: none

    subroutine zadd(a,b,c,n) ! in :add:add.f 
       double complex dimension(n) :: a 
       double complex dimension(n) :: b 
       double complex intent(out),dimension(n) :: c 
       integer intent(hide),depend(a) :: n=len(a) 
    end subroutine zadd

The intent directive, intent(out) is used to tell f2py that ``c`` is
an output variable and should be created by the interface before being
passed to the underlying code. The intent(hide) directive tells f2py
to not allow the user to specify the variable, ``n``, but instead to
get it from the size of ``a``. The depend( ``a`` ) directive is
necessary to tell f2py that the value of n depends on the input a (so
that it won't try to create the variable n until the variable a is
created). 

The new interface has docstring:

    >>> print add.zadd.__doc__
    zadd - Function signature: 
      c = zadd(a,b) 
    Required arguments: 
      a : input rank-1 array('D') with bounds (n) 
      b : input rank-1 array('D') with bounds (n) 
    Return objects: 
      c : rank-1 array('D') with bounds (n) 

Now, the function can be called in a much more robust way: 

    >>> add.zadd([1,2,3],[4,5,6])
    array([ 5.+0.j,  7.+0.j,  9.+0.j])

Notice the automatic conversion to the correct format that occurred. 


Inserting directives in Fortran source
--------------------------------------

The nice interface can also be generated automatically by placing the
variable directives as special comments in the original fortran code.
Thus, if I modify the source code to contain:

.. code-block:: none

    C
          SUBROUTINE ZADD(A,B,C,N)
    C
    CF2PY INTENT(OUT) :: C
    CF2PY INTENT(HIDE) :: N
    CF2PY DOUBLE COMPLEX :: A(N)
    CF2PY DOUBLE COMPLEX :: B(N)
    CF2PY DOUBLE COMPLEX :: C(N)
          DOUBLE COMPLEX A(*)
          DOUBLE COMPLEX B(*)
          DOUBLE COMPLEX C(*)
          INTEGER N
          DO 20 J = 1, N
             C(J) = A(J) + B(J)
     20   CONTINUE
          END

Then, I can compile the extension module using::

    f2py -c -m add add.f

The resulting signature for the function add.zadd is exactly the same
one that was created previously. If the original source code had
contained A(N) instead of A(\*) and so forth with B and C, then I
could obtain (nearly) the same interface simply by placing the
INTENT(OUT) :: C comment line in the source code. The only difference
is that N would be an optional input that would default to the length
of A. 


A filtering example
-------------------

For comparison with the other methods to be discussed. Here is another
example of a function that filters a two-dimensional array of double
precision floating-point numbers using a fixed averaging filter. The
advantage of using Fortran to index into multi-dimensional arrays
should be clear from this example. 

.. code-block:: none

          SUBROUTINE DFILTER2D(A,B,M,N)
    C
          DOUBLE PRECISION A(M,N)
          DOUBLE PRECISION B(M,N)
          INTEGER N, M
    CF2PY INTENT(OUT) :: B
    CF2PY INTENT(HIDE) :: N
    CF2PY INTENT(HIDE) :: M
          DO 20 I = 2,M-1
             DO 40 J=2,N-1
                B(I,J) = A(I,J) + 
         $           (A(I-1,J)+A(I+1,J) +
         $            A(I,J-1)+A(I,J+1) )*0.5D0 +
         $           (A(I-1,J-1) + A(I-1,J+1) +
         $            A(I+1,J-1) + A(I+1,J+1))*0.25D0
     40      CONTINUE
     20   CONTINUE
          END

This code can be compiled and linked into an extension module named
filter using::

    f2py -c -m filter filter.f

This will produce an extension module named filter.so in the current
directory with a method named dfilter2d that returns a filtered
version of the input. 


Calling f2py from Python
------------------------

The f2py program is written in Python and can be run from inside your
module. This provides a facility that is somewhat similar to the use
of weave.ext_tools described below. An example of the final interface
executed using Python code is:

.. code-block:: python

