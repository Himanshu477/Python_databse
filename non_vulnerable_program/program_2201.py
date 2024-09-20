  import numpy as np

  # A normal ndarray, that owns its own data
  arr = np.zeros((4,))
  # In this case, base is None
  assert arr.base is None
  # We take a view
  v1 = arr[1:]
  # base now points to the array that it derived from
  assert v1.base is arr
  # Take a view of a view
  v2 = v1[1:]
  # base points to the view it derived from
  assert v2.base is v1

The assertions all succeed in this case.  In general, if the array
owns its own memory, as for ``arr`` in this case, then ``arr.base``
will be None - there are some exceptions to this - see the numpy book
for more details.

The ``base`` attribute is useful in being able to tell whether we have
a view or the original array.  This in turn can be useful if we need
to know whether or not to do some specific cleanup when the subclassed
array is deleted.  For example, we may only want to do the cleanup if
the original array is deleted, but not the views.  For an example of
how this can work, have a look at the ``memmap`` class in
``numpy.core``.

"""


__all__ = ['filter2d']

