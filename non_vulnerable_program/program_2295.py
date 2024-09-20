        from numpy.distutils.core import setup
        setup(**configuration(top_path='').todict())

Installation of the new package is easy using::

    python setup.py install

assuming you have the proper permissions to write to the main site-
packages directory for the version of Python you are using. For the
resulting package to work, you need to create a file named __init__.py
(in the same directory as add.pyf). Notice the extension module is
defined entirely in terms of the "add.pyf" and "add.f" files. The
conversion of the .pyf file to a .c file is handled by numpy.disutils.


Conclusion
----------

The interface definition file (.pyf) is how you can fine-tune the
interface between Python and Fortran. There is decent documentation
for f2py found in the numpy/f2py/docs directory where-ever NumPy is
installed on your system (usually under site-packages). There is also
more information on using f2py (including how to use it to wrap C
codes) at http://www.scipy.org/Cookbook under the "Using NumPy with
Other Languages" heading.

The f2py method of linking compiled code is currently the most
sophisticated and integrated approach. It allows clean separation of
Python with compiled code while still allowing for separate
distribution of the extension module. The only draw-back is that it
requires the existence of a Fortran compiler in order for a user to
install the code. However, with the existence of the free-compilers
g77, gfortran, and g95, as well as high-quality commerical compilers,
this restriction is not particularly onerous. In my opinion, Fortran
is still the easiest way to write fast and clear code for scientific
computing. It handles complex numbers, and multi-dimensional indexing
in the most straightforward way. Be aware, however, that some Fortran
compilers will not be able to optimize code as well as good hand-
written C-code.

.. index::
   single: f2py


weave
=====

Weave is a scipy package that can be used to automate the process of
extending Python with C/C++ code. It can be used to speed up
evaluation of an array expression that would otherwise create
temporary variables, to directly "inline" C/C++ code into Python, or
to create a fully-named extension module.  You must either install
scipy or get the weave package separately and install it using the
standard python setup.py install. You must also have a C/C++-compiler
installed and useable by Python distutils in order to use weave.

.. index::
   single: weave

Somewhat dated, but still useful documentation for weave can be found
at the link http://www.scipy/Weave. There are also many examples found
in the examples directory which is installed under the weave directory
in the place where weave is installed on your system.


Speed up code involving arrays (also see scipy.numexpr)
-------------------------------------------------------

This is the easiest way to use weave and requires minimal changes to
your Python code. It involves placing quotes around the expression of
interest and calling weave.blitz. Weave will parse the code and
generate C++ code using Blitz C++ arrays. It will then compile the
code and catalog the shared library so that the next time this exact
string is asked for (and the array types are the same), the already-
compiled shared library will be loaded and used. Because Blitz makes
extensive use of C++ templating, it can take a long time to compile
the first time. After that, however, the code should evaluate more
quickly than the equivalent NumPy expression. This is especially true
if your array sizes are large and the expression would require NumPy
to create several temporaries. Only expressions involving basic
arithmetic operations and basic array slicing can be converted to
Blitz C++ code.

For example, consider the expression::

    d = 4*a + 5*a*b + 6*b*c

where a, b, and c are all arrays of the same type and shape. When the
data-type is double-precision and the size is 1000x1000, this
expression takes about 0.5 seconds to compute on an 1.1Ghz AMD Athlon
machine. When this expression is executed instead using blitz:

.. code-block:: python

    d = empty(a.shape, 'd'); weave.blitz(expr)

execution time is only about 0.20 seconds (about 0.14 seconds spent in
weave and the rest in allocating space for d). Thus, we've sped up the
code by a factor of 2 using only a simnple command (weave.blitz). Your
mileage may vary, but factors of 2-8 speed-ups are possible with this
very simple technique.

If you are interested in using weave in this way, then you should also
look at scipy.numexpr which is another similar way to speed up
expressions by eliminating the need for temporary variables. Using
numexpr does not require a C/C++ compiler.


Inline C-code
-------------

Probably the most widely-used method of employing weave is to
"in-line" C/C++ code into Python in order to speed up a time-critical
section of Python code. In this method of using weave, you define a
string containing useful C-code and then pass it to the function
**weave.inline** ( ``code_string``, ``variables`` ), where
code_string is a string of valid C/C++ code and variables is a list of
variables that should be passed in from Python. The C/C++ code should
refer to the variables with the same names as they are defined with in
Python. If weave.line should return anything the the special value
return_val should be set to whatever object should be returned. The
following example shows how to use weave on basic Python objects:

.. code-block:: python

    code = r"""
    int i;
    py::tuple results(2);
    for (i=0; i<a.length(); i++) {
         a[i] = i;
    }
    results[0] = 3.0;
    results[1] = 4.0;
    return_val = results;
    """
    a = [None]*10
    res = weave.inline(code,['a'])

The C++ code shown in the code string uses the name 'a' to refer to
the Python list that is passed in. Because the Python List is a
mutable type, the elements of the list itself are modified by the C++
code. A set of C++ classes are used to access Python objects using
simple syntax.

The main advantage of using C-code, however, is to speed up processing
on an array of data. Accessing a NumPy array in C++ code using weave,
depends on what kind of type converter is chosen in going from NumPy
arrays to C++ code. The default converter creates 5 variables for the
C-code for every NumPy array passed in to weave.inline. The following
table shows these variables which can all be used in the C++ code. The
table assumes that ``myvar`` is the name of the array in Python with
data-type {dtype} (i.e.  float64, float32, int8, etc.)

===========  ==============  =========================================
Variable     Type            Contents
===========  ==============  =========================================
myvar        {dtype}*        Pointer to the first element of the array
Nmyvar       npy_intp*       A pointer to the dimensions array
Smyvar       npy_intp*       A pointer to the strides array
Dmyvar       int             The number of dimensions
myvar_array  PyArrayObject*  The entire structure for the array
===========  ==============  =========================================

The in-lined code can contain references to any of these variables as
well as to the standard macros MYVAR1(i), MYVAR2(i,j), MYVAR3(i,j,k),
and MYVAR4(i,j,k,l). These name-based macros (they are the Python name
capitalized followed by the number of dimensions needed) will de-
reference the memory for the array at the given location with no error
checking (be-sure to use the correct macro and ensure the array is
aligned and in correct byte-swap order in order to get useful
results). The following code shows how you might use these variables
and macros to code a loop in C that computes a simple 2-d weighted
averaging filter.

.. code-block:: c++

    int i,j;
    for(i=1;i<Na[0]-1;i++) {
       for(j=1;j<Na[1]-1;j++) {
           B2(i,j) = A2(i,j) + (A2(i-1,j) +
                     A2(i+1,j)+A2(i,j-1)
                     + A2(i,j+1))*0.5
                     + (A2(i-1,j-1)
                     + A2(i-1,j+1)
                     + A2(i+1,j-1)
                     + A2(i+1,j+1))*0.25
       }
    }

The above code doesn't have any error checking and so could fail with
a Python crash if, ``a`` had the wrong number of dimensions, or ``b``
did not have the same shape as ``a``. However, it could be placed
inside a standard Python function with the necessary error checking to
produce a robust but fast subroutine.

One final note about weave.inline: if you have additional code you
want to include in the final extension module such as supporting
function calls, include statments, etc. you can pass this code in as a
string using the keyword support_code: ``weave.inline(code, variables,
support_code=support)``. If you need the extension module to link
against an additional library then you can also pass in
distutils-style keyword arguments such as library_dirs, libraries,
and/or runtime_library_dirs which point to the appropriate libraries
and directories.

Simplify creation of an extension module
----------------------------------------

The inline function creates one extension module for each function to-
be inlined. It also generates a lot of intermediate code that is
duplicated for each extension module. If you have several related
codes to execute in C, it would be better to make them all separate
functions in a single extension module with multiple functions. You
can also use the tools weave provides to produce this larger extension
module. In fact, the weave.inline function just uses these more
general tools to do its work.

The approach is to:

1. construct a extension module object using
   ext_tools.ext_module(``module_name``);

2. create function objects using ext_tools.ext_function(``func_name``,
   ``code``, ``variables``);

3. (optional) add support code to the function using the
   .customize.add_support_code( ``support_code`` ) method of the
   function object;

4. add the functions to the extension module object using the
   .add_function(``func``) method;

5. when all the functions are added, compile the extension with its
   .compile() method.

Several examples are available in the examples directory where weave
is installed on your system. Look particularly at ramp2.py,
increment_example.py and fibonacii.py


Conclusion
----------

Weave is a useful tool for quickly routines in C/C++ and linking them
into Python. It's caching-mechanism allows for on-the-fly compilation
which makes it particularly attractive for in-house code. Because of
the requirement that the user have a C++-compiler, it can be difficult
(but not impossible) to distribute a package that uses weave to other
users who don't have a compiler installed. Of course, weave could be
used to construct an extension module which is then distributed in the
normal way *(* using a setup.py file). While you can use weave to
build larger extension modules with many methods, creating methods
with a variable- number of arguments is not possible. Thus, for a more
sophisticated module, you will still probably want a Python-layer that
calls the weave-produced extension.

.. index::
   single: weave


Pyrex
=====

Pyrex is a way to write C-extension modules using Python-like syntax.
It is an interesting way to generate extension modules that is growing
in popularity, particularly among people who have rusty or non-
existent C-skills. It does require the user to write the "interface"
code and so is more time-consuming than SWIG or f2py if you are trying
to interface to a large library of code. However, if you are writing
an extension module that will include quite a bit of your own
algorithmic code, as well, then Pyrex is a good match. A big weakness
perhaps is the inability to easily and quickly access the elements of
a multidimensional array.

.. index::
   single: pyrex

Notice that Pyrex is an extension-module generator only. Unlike weave
or f2py, it includes no automatic facility for compiling and linking
the extension module (which must be done in the usual fashion). It
does provide a modified distutils class called build_ext which lets
you build an extension module from a .pyx source. Thus, you could
write in a setup.py file:

.. code-block:: python

