  import Numeric, dotblas
  Numeric.dot = dotblas.dot

You can also just add the following line at the end of your ``Numeric.py``
to globally use the optimized `dot` function::

