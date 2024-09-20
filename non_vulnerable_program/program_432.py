from ppimport import ppimport, ppimport_attr

class _TypeNamespace:
    """Numeric compatible type aliases for use with extension functions."""
    Int8          = Int8
    UInt8         = UInt8
    Int16         = Int16
    UInt16        = UInt16
    Int32         = Int32
    UInt32        = UInt32
    Float32       = Float32
    Float64       = Float64
    Complex32     = Complex32
    Complex64     = Complex64

nx = _TypeNamespace()

# inf is useful for testing infinities in results of array divisions
# (which don't raise exceptions)

inf = infty = Infinity = (array([1])/0.0)[0]

# The following import statements are equivalent to
#
#   from Matrix import Matrix as mat
#
# but avoids expensive LinearAlgebra import when
# Matrix is not used.
#
LinearAlgebra = ppimport('LinearAlgebra')
inverse = ppimport_attr(LinearAlgebra, 'inverse')
eigenvectors = ppimport_attr(LinearAlgebra, 'eigenvectors')
Matrix = mat = ppimport_attr(ppimport('Matrix'), 'Matrix')
fft = ppimport_attr(ppimport('FFT'), 'fft')
RandomArray =  ppimport('RandomArray')
MLab = ppimport('MLab')

NUMERIX_HEADER = "Numeric/arrayobject.h"

#
# Force numerix to use scipy_base.fastumath instead of numerix.umath.
#
