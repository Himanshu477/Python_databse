import re
import numpy as np
import os

names = re.compile(r'r\d+\s[|]\s(.*)\s[|]\s200')

def get_count(filename, repo):
    mystr = open(filename).read()
    result = names.findall(mystr)
    u = np.unique(result)
    count = [(x,result.count(x),repo) for x in u]
    return count
    

command = 'svn log -l 2300 > output.txt'
os.chdir('..')
os.system(command)

count = get_count('output.txt', 'NumPy')


os.chdir('../scipy')
os.system(command)

count.extend(get_count('output.txt', 'SciPy'))

os.chdir('../scikits')
os.system(command)
count.extend(get_count('output.txt', 'SciKits'))
count.sort()

             

print "** SciPy and NumPy **"
print "====================="
for val in count:
    print val


                   


"""
========================
Broadcasting over arrays
========================

The term broadcasting describes how numpy treats arrays with different
shapes during arithmetic operations. Subject to certain constraints,
the smaller array is "broadcast" across the larger array so that they
have compatible shapes. Broadcasting provides a means of vectorizing
array operations so that looping occurs in C instead of Python. It does
this without making needless copies of data and usually leads to
efficient algorithm implementations. There are, however, cases where
broadcasting is a bad idea because it leads to inefficient use of memory
that slows computation.

NumPy operations are usually done on pairs of arrays on an
element-by-element basis.  In the simplest case, the two arrays must
have exactly the same shape, as in the following example:

  >>> a = np.array([1.0, 2.0, 3.0])
  >>> b = np.array([2.0, 2.0, 2.0])
  >>> a * b
  array([ 2.,  4.,  6.])

NumPy's broadcasting rule relaxes this constraint when the arrays'
shapes meet certain constraints. The simplest broadcasting example occurs
when an array and a scalar value are combined in an operation:

>>> a = np.array([1.0, 2.0, 3.0])
>>> b = 2.0
>>> a * b
array([ 2.,  4.,  6.])

The result is equivalent to the previous example where ``b`` was an array.
We can think of the scalar ``b`` being *stretched* during the arithmetic
operation into an array with the same shape as ``a``. The new elements in
``b`` are simply copies of the original scalar. The stretching analogy is
only conceptual.  NumPy is smart enough to use the original scalar value
without actually making copies, so that broadcasting operations are as
memory and computationally efficient as possible.

The code in the second example is more efficient than that in the first
because broadcasting moves less memory around during the multiplication
(``b`` is a scalar rather than an array).

General Broadcasting Rules
==========================
When operating on two arrays, NumPy compares their shapes element-wise.
It starts with the trailing dimensions, and works its way forward.  Two
dimensions are compatible when

1) they are equal, or
2) one of them is 1

If these conditions are not met, a
``ValueError: frames are not aligned`` exception is thrown, indicating that
the arrays have incompatible shapes. The size of the resulting array
is the maximum size along each dimension of the input arrays.

Arrays do not need to have the same *number* of dimensions.  For example,
if you have a ``256x256x3`` array of RGB values, and you want to scale
each color in the image by a different value, you can multiply the image
by a one-dimensional array with 3 values. Lining up the sizes of the
trailing axes of these arrays according to the broadcast rules, shows that
they are compatible::

  Image  (3d array): 256 x 256 x 3
  Scale  (1d array):             3
  Result (3d array): 256 x 256 x 3

When either of the dimensions compared is one, the larger of the two is
used.  In other words, the smaller of two axes is stretched or "copied"
to match the other.

In the following example, both the ``A`` and ``B`` arrays have axes with
length one that are expanded to a larger size during the broadcast
operation::

  A      (4d array):  8 x 1 x 6 x 1
  B      (3d array):      7 x 1 x 5
  Result (4d array):  8 x 7 x 6 x 5

Here are some more examples::

  A      (2d array):  5 x 4
  B      (1d array):      1
  Result (2d array):  5 x 4

  A      (2d array):  5 x 4
  B      (1d array):      4
  Result (2d array):  5 x 4

  A      (3d array):  15 x 3 x 5
  B      (3d array):  15 x 1 x 5
  Result (3d array):  15 x 3 x 5

  A      (3d array):  15 x 3 x 5
  B      (2d array):       3 x 5
  Result (3d array):  15 x 3 x 5

  A      (3d array):  15 x 3 x 5
  B      (2d array):       3 x 1
  Result (3d array):  15 x 3 x 5

Here are examples of shapes that do not broadcast::

  A      (1d array):  3
  B      (1d array):  4 # trailing dimensions do not match

  A      (2d array):      2 x 1
  B      (3d array):  8 x 4 x 3 # second from last dimensions mismatched

An example of broadcasting in practice::

 >>> x = np.arange(4)
 >>> xx = x.reshape(4,1)
 >>> y = np.ones(5)
 >>> z = np.ones((3,4))

 >>> x.shape
 (4,)

 >>> y.shape
 (5,)

 >>> x + y
 <type 'exceptions.ValueError'>: shape mismatch: objects cannot be broadcast to a single shape

 >>> xx.shape
 (4, 1)

 >>> y.shape
 (5,)

 >>> (xx + y).shape
 (4, 5)

 >>> xx + y
 array([[ 1.,  1.,  1.,  1.,  1.],
        [ 2.,  2.,  2.,  2.,  2.],
        [ 3.,  3.,  3.,  3.,  3.],
        [ 4.,  4.,  4.,  4.,  4.]])

 >>> x.shape
 (4,)

 >>> z.shape
 (3, 4)

 >>> (x + z).shape
 (3, 4)

 >>> x + z
 array([[ 1.,  2.,  3.,  4.],
        [ 1.,  2.,  3.,  4.],
        [ 1.,  2.,  3.,  4.]])

Broadcasting provides a convenient way of taking the outer product (or
any other outer operation) of two arrays. The following example shows an
outer addition operation of two 1-d arrays::

  >>> a = np.array([0.0, 10.0, 20.0, 30.0])
  >>> b = np.array([1.0, 2.0, 3.0])
  >>> a[:, np.newaxis] + b
  array([[  1.,   2.,   3.],
         [ 11.,  12.,  13.],
         [ 21.,  22.,  23.],
         [ 31.,  32.,  33.]])

Here the ``newaxis`` index operator inserts a new axis into ``a``,
making it a two-dimensional ``4x1`` array.  Combining the ``4x1`` array
with ``b``, which has shape ``(3,)``, yields a ``4x3`` array.

See `this article <http://www.scipy.org/EricsBroadcastingDoc>`_
for illustrations of broadcasting concepts.

"""


"""
=====================================
Structured Arrays (aka Record Arrays)
=====================================

Introduction
============

Numpy provides powerful capabilities to create arrays of structs or records.
These arrays permit one to manipulate the data by the structs or by fields of
the struct. A simple example will show what is meant.: ::

 >>> x = np.zeros((2,),dtype=('i4,f4,a10'))
 >>> x[:] = [(1,2.,'Hello'),(2,3.,"World")]
 >>> x
 array([(1, 2.0, 'Hello'), (2, 3.0, 'World')],
      dtype=[('f0', '>i4'), ('f1', '>f4'), ('f2', '|S10')])

Here we have created a one-dimensional array of length 2. Each element of
this array is a record that contains three items, a 32-bit integer, a 32-bit
float, and a string of length 10 or less. If we index this array at the second
position we get the second record: ::

 >>> x[1]
 (2,3.,"World")

Conveniently, one can access any field of the array by indexing using the
string that names that field. In this case the fields have received the
default names 'f0', 'f1' and 'f2'.

 >>> y = x['f1']
 >>> y
 array([ 2.,  3.], dtype=float32)
 >>> y[:] = 2*y
 >>> y
 array([ 4.,  6.], dtype=float32)
 >>> x
 array([(1, 4.0, 'Hello'), (2, 6.0, 'World')],
       dtype=[('f0', '>i4'), ('f1', '>f4'), ('f2', '|S10')])

In these examples, y is a simple float array consisting of the 2nd field
in the record. But, rather than being a copy of the data in the structured
array, it is a view, i.e., it shares exactly the same memory locations.
Thus, when we updated this array by doubling its values, the structured
array shows the corresponding values as doubled as well. Likewise, if one
changes the record, the field view also changes: ::

 >>> x[1] = (-1,-1.,"Master")
 >>> x
 array([(1, 4.0, 'Hello'), (-1, -1.0, 'Master')],
       dtype=[('f0', '>i4'), ('f1', '>f4'), ('f2', '|S10')])
 >>> y
 array([ 4., -1.], dtype=float32)

Defining Structured Arrays
==========================

One defines a structured array through the dtype object.  There are
**several** alternative ways to define the fields of a record.  Some of
these variants provide backward compatibility with Numeric, numarray, or
another module, and should not be used except for such purposes. These
will be so noted. One specifies record structure in
one of four alternative ways, using an argument (as supplied to a dtype
function keyword or a dtype object constructor itself).  This
argument must be one of the following: 1) string, 2) tuple, 3) list, or
4) dictionary.  Each of these is briefly described below.

1) String argument (as used in the above examples).
In this case, the constructor expects a comma-separated list of type
specifiers, optionally with extra shape information.
The type specifiers can take 4 different forms: ::

  a) b1, i1, i2, i4, i8, u1, u2, u4, u8, f4, f8, c8, c16, a<n>
     (representing bytes, ints, unsigned ints, floats, complex and
      fixed length strings of specified byte lengths)
  b) int8,...,uint8,...,float32, float64, complex64, complex128
     (this time with bit sizes)
  c) older Numeric/numarray type specifications (e.g. Float32).
     Don't use these in new code!
  d) Single character type specifiers (e.g H for unsigned short ints).
     Avoid using these unless you must. Details can be found in the
     Numpy book

These different styles can be mixed within the same string (but why would you
want to do that?). Furthermore, each type specifier can be prefixed
with a repetition number, or a shape. In these cases an array
element is created, i.e., an array within a record. That array
is still referred to as a single field. An example: ::

 >>> x = np.zeros(3, dtype='3int8, float32, (2,3)float64')
 >>> x
 array([([0, 0, 0], 0.0, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        ([0, 0, 0], 0.0, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        ([0, 0, 0], 0.0, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])],
       dtype=[('f0', '|i1', 3), ('f1', '>f4'), ('f2', '>f8', (2, 3))])

By using strings to define the record structure, it precludes being
able to name the fields in the original definition. The names can
be changed as shown later, however.

2) Tuple argument: The only relevant tuple case that applies to record
structures is when a structure is mapped to an existing data type. This
is done by pairing in a tuple, the existing data type with a matching
dtype definition (using any of the variants being described here). As
an example (using a definition using a list, so see 3) for further
details): ::

 >>> x = zeros(3, dtype=('i4',[('r','u1'), ('g','u1'), ('b','u1'), ('a','u1')]))
 >>> x
 array([0, 0, 0])
 >>> x['r']
 array([0, 0, 0], dtype=uint8)

In this case, an array is produced that looks and acts like a simple int32 array,
but also has definitions for fields that use only one byte of the int32 (a bit
like Fortran equivalencing).

3) List argument: In this case the record structure is defined with a list of
tuples. Each tuple has 2 or 3 elements specifying: 1) The name of the field
('' is permitted), 2) the type of the field, and 3) the shape (optional).
For example:

 >>> x = np.zeros(3, dtype=[('x','f4'),('y',np.float32),('value','f4',(2,2))])
 >>> x
 array([(0.0, 0.0, [[0.0, 0.0], [0.0, 0.0]]),
        (0.0, 0.0, [[0.0, 0.0], [0.0, 0.0]]),
        (0.0, 0.0, [[0.0, 0.0], [0.0, 0.0]])],
       dtype=[('x', '>f4'), ('y', '>f4'), ('value', '>f4', (2, 2))])

4) Dictionary argument: two different forms are permitted. The first consists
of a dictionary with two required keys ('names' and 'formats'), each having an
equal sized list of values. The format list contains any type/shape specifier
allowed in other contexts. The names must be strings. There are two optional
keys: 'offsets' and 'titles'. Each must be a correspondingly matching list to
the required two where offsets contain integer offsets for each field, and
titles are objects containing metadata for each field (these do not have
to be strings), where the value of None is permitted. As an example: ::

 >>> x = np.zeros(3, dtype={'names':['col1', 'col2'], 'formats':['i4','f4']})
 >>> x
 array([(0, 0.0), (0, 0.0), (0, 0.0)],
       dtype=[('col1', '>i4'), ('col2', '>f4')])

The other dictionary form permitted is a dictionary of name keys with tuple
values specifying type, offset, and an optional title.

 >>> x = np.zeros(3, dtype={'col1':('i1',0,'title 1'), 'col2':('f4',1,'title 2')})
 array([(0, 0.0), (0, 0.0), (0, 0.0)],
       dtype=[(('title 1', 'col1'), '|i1'), (('title 2', 'col2'), '>f4')])

Accessing and modifying field names
===================================

The field names are an attribute of the dtype object defining the record structure.
For the last example: ::

 >>> x.dtype.names
 ('col1', 'col2')
 >>> x.dtype.names = ('x', 'y')
 >>> x
 array([(0, 0.0), (0, 0.0), (0, 0.0)],
      dtype=[(('title 1', 'x'), '|i1'), (('title 2', 'y'), '>f4')])
 >>> x.dtype.names = ('x', 'y', 'z') # wrong number of names
 <type 'exceptions.ValueError'>: must replace all names at once with a sequence of length 2

Accessing field titles
====================================

The field titles provide a standard place to put associated info for fields.
They do not have to be strings.

 >>> x.dtype.fields['x'][2]
 'title 1'

"""


__all__ = ['matrix', 'bmat', 'mat', 'asmatrix']

