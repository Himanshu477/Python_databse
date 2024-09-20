from machar import MachAr
import numeric
import numerictypes as ntypes
from numeric import array

def _frz(a):
    """fix rank-0 --> rank-1"""
    if a.ndim == 0: a.shape = (1,)
    return a

_convert_to_float = {
    ntypes.csingle: ntypes.single,
    ntypes.complex_: ntypes.float_,
    ntypes.clongfloat: ntypes.longfloat
    }

class finfo(object):
    """
    finfo(dtype)

    Machine limits for floating point types.

    Attributes
    ----------
    eps : floating point number of the appropriate type
        The smallest representable number such that ``1.0 + eps != 1.0``.
    epsneg : floating point number of the appropriate type
        The smallest representable number such that ``1.0 - epsneg != 1.0``.
    iexp : int
        The number of bits in the exponent portion of the floating point
        representation.
    machar : MachAr
        The object which calculated these parameters and holds more detailed
        information.
    machep : int
        The exponent that yields ``eps``.
    max : floating point number of the appropriate type
        The largest representable number.
    maxexp : int
        The smallest positive power of the base (2) that causes overflow.
    min : floating point number of the appropriate type
        The smallest representable number, typically ``-max``.
    minexp : int
        The most negative power of the base (2) consistent with there being
        no leading 0s in the mantissa.
    negep : int
        The exponent that yields ``epsneg``.
    nexp : int
        The number of bits in the exponent including its sign and bias.
    nmant : int
        The number of bits in the mantissa.
    precision : int
        The approximate number of decimal digits to which this kind of float
        is precise.
    resolution : floating point number of the appropriate type
        The approximate decimal resolution of this type, i.e.
        ``10**-precision``.
    tiny : floating point number of the appropriate type
        The smallest-magnitude usable number.

    Parameters
    ----------
    dtype : floating point type, dtype, or instance
        The kind of floating point data type to get information about.

    See Also
    --------
    numpy.lib.machar.MachAr :
        The implementation of the tests that produce this information.
    iinfo :
        The equivalent for integer data types.

    Notes
    -----
    For developers of numpy: do not instantiate this at the module level. The
    initial calculation of these parameters is expensive and negatively impacts
