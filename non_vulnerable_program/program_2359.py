from numpy.core.fromnumeric import any
from numpy.core.numeric import seterr

# Need to speed this up...especially for longfloat

class MachAr(object):
    """
    Diagnosing machine parameters.

    Attributes
    ----------
    ibeta : int
        Radix in which numbers are represented.
    it : int
        Number of base-`ibeta` digits in the floating point mantissa M.
    machep : int
        Exponent of the smallest (most negative) power of `ibeta` that,
        added to 1.0, gives something different from 1.0
    eps : float
        Floating-point number ``beta**machep`` (floating point precision)
    negep : int
        Exponent of the smallest power of `ibeta` that, substracted
