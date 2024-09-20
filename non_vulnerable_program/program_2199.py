  import numpy as np

  class InfoArray(np.ndarray):

      def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
        strides=None, order=None, info=None):
      # Create the ndarray instance of our type, given the usual
      # input arguments.  This will call the standard ndarray
      # constructor, but return an object of our type
      obj = np.ndarray.__new__(subtype, shape, dtype, buffer, offset, strides,
               order)
      # add the new attribute to the created instance
      obj.info = info
      # Finally, we must return the newly created object:
      return obj

      def __array_finalize__(self,obj):
      # reset the attribute from passed original object
      self.info = getattr(obj, 'info', None)
      # We do not need to return anything

  obj = InfoArray(shape=(3,), info='information')
  print type(obj)
  print obj.info
  v = obj[1:]
  print type(v)
  print v.info

which gives::

  <class '__main__.InfoArray'>
  information
  <class '__main__.InfoArray'>
  information

This class isn't very useful, because it has the same constructor as
the bare ndarray object, including passing in buffers and shapes and
so on.   We would probably prefer to be able to take an already formed
ndarray from the usual numpy calls to ``np.array`` and return an
object.

Slightly more realistic example - attribute added to existing array
-------------------------------------------------------------------
Here is a class (with thanks to PierreGM for the original example,
that takes array that already exists, casts as our type, and adds an
extra attribute::

