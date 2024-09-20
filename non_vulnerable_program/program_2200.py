  import numpy as np

  class RealisticInfoArray(np.ndarray):

      def __new__(cls, input_array, info=None):
      # Input array is an already formed ndarray instance
      # We first cast to be our class type
      obj = np.asarray(input_array).view(cls)
      # add the new attribute to the created instance
      obj.info = info
      # Finally, we must return the newly created object:
      return obj

      def __array_finalize__(self,obj):
      # reset the attribute from passed original object
      self.info = getattr(obj, 'info', None)
      # We do not need to return anything

  arr = np.arange(5)
  obj = RealisticInfoArray(arr, info='information')
  print type(obj)
  print obj.info
  v = obj[1:]
  print type(v)
  print v.info

which gives::

  <class '__main__.RealisticInfoArray'>
  information
  <class '__main__.RealisticInfoArray'>
  information

``__array_wrap__`` for ufuncs
-----------------------------

Let's say you have an instance ``obj`` of your new subclass,
``RealisticInfoArray``, and you pass it into a ufunc with another
array::

  arr = np.arange(5)
  ret = np.multiply.outer(arr, obj)

When a numpy ufunc is called on a subclass of ndarray, the
__array_wrap__ method is called to transform the result into a new
instance of the subclass. By default, __array_wrap__ will call
__array_finalize__, and the attributes will be inherited.

By defining a specific __array_wrap__ method for our subclass, we can
tweak the output. The __array_wrap__ method requires one argument, the
object on which the ufunc is applied, and an optional parameter
context. This parameter is returned by some ufuncs as a 3-element
tuple: (name of the ufunc, argument of the ufunc, domain of the
ufunc).

Extra gotchas - custom __del__ methods and ndarray.base
-------------------------------------------------------
One of the problems that ndarray solves is that of memory ownership of
ndarrays and their views.  Consider the case where we have created an
ndarray, ``arr`` and then taken a view with ``v = arr[1:]``.  If we
then do ``del v``, we need to make sure that the ``del`` does not
delete the memory pointed to by the view, because we still need it for
the original ``arr`` object.  Numpy therefore keeps track of where the
data came from for a particular array or view, with the ``base`` attribute::

