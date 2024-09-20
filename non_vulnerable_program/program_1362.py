from numpy.core.numeric import array
from numpy.core.oldnumeric import any

# Need to speed this up...especially for longfloat

class MachAr(object):
    """Diagnosing machine parameters.

    The following attributes are available:

    ibeta  - radix in which numbers are represented
    it     - number of base-ibeta digits in the floating point mantissa M
    machep - exponent of the smallest (most negative) power of ibeta that,
             added to 1.0,
             gives something different from 1.0
    eps    - floating-point number beta**machep (floating point precision)
    negep  - exponent of the smallest power of ibeta that, substracted
