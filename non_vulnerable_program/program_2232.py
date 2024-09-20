import statement are supported. It also shows use of the NumPy C-API
to construct NumPy arrays from arbitrary input objects. The array c is
created using PyArray_SimpleNew. Then the c-array is filled by
addition. Casting to a particiular data-type is accomplished using
<cast \*>. Pointers are de-referenced with bracket notation and
members of structures are accessed using '.' notation even if the
object is techinically a pointer to a structure. The use of the
special for loop construct ensures that the underlying code will have
a similar C-loop so the addition calculation will proceed quickly.
Notice that we have not checked for NULL after calling to the C-API
--- a cardinal sin when writing C-code. For routines that return
Python objects, Pyrex inserts the checks for NULL into the C-code for
you and returns with failure if need be. There is also a way to get
Pyrex to automatically check for exceptions when you call functions
that don't return Python objects. See the documentation of Pyrex for
details. 


Pyrex-filter
------------

The two-dimensional example we created using weave is a bit uglierto
implement in Pyrex because two-dimensional indexing using Pyrex is not
as simple. But, it is straightforward (and possibly faster because of
pre-computed indices). Here is the Pyrex-file I named image.pyx. 

.. code-block:: none

    cimport c_numpy
