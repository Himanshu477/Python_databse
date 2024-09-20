import Numeric
import _dotblas
import multiarray

def dot(a, b):
    """returns matrix-multiplication between a and b.
    The product-sum is over the last dimension of a and the
    second-to-last dimension of b.

    NB: No conjugation of complex arguments is performed.

    This version uses the BLAS optimized routines where possible.
    """
    try:
        return _dotblas.matrixproduct(a, b)
    except:
        try:
            return multiarray.matrixproduct(a, b)
        except TypeError,detail:
            if multiarray.array(a).shape == () or multiarray.array(b).shape == ():
                return a*b
            else:
                raise TypeError, detail or "invalid types for matrixproduct"

def innerproduct(a, b):
    """returns inner product between a and b.
    The product-sum is over the last dimension of a and b.

    NB: No conjugation of complex arguments is performed.

    This version uses the BLAS optimized routines where possible.
    """
    try:
        return _dotblas.innerproduct(a, b)
    except TypeError:
        try:
            return multiarray.innerproduct(a, b)
        except TypeError,detail:
            if multiarray.array(a).shape == () or multiarray.array(b).shape == ():
                return a*b
            else:
                raise TypeError, detail or "invalid types for innerproduct"

def _is_vector(x):
    return Numeric.rank(x) == 1 or \
           Numeric.rank(x) == 2 and min(Numeric.shape(x)) == 1
matrixmultiply = dot


def vdot(a, b):
    """Returns the dot product of 2 vectors (or anything that can be made into
       a vector). NB: this is not the same as `dot`, as it takes the conjugate
       of its first argument if complex and always returns a scalar."""
    try:
        return _dotblas.vdot(Numeric.ravel(a), Numeric.ravel(b))
    # in case we get an integer Value
    except TypeError:
        return multiarray.matrixproduct(multiarray.array(a).flat,
                                        multiarray.array(b).flat)

__all__ = 'dot innerproduct vdot'.split()


"""
Discrete Fourier Transforms - FFT.py 

The underlying code for these functions is an f2c translated and modified
version of the FFTPACK routines.

fft(a, n=None, axis=-1) 
inverse_fft(a, n=None, axis=-1) 
real_fft(a, n=None, axis=-1) 
inverse_real_fft(a, n=None, axis=-1)
hermite_fft(a, n=None, axis=-1)
inverse_hermite_fft(a, n=None, axis=-1)
fftnd(a, s=None, axes=None)
inverse_fftnd(a, s=None, axes=None)
real_fftnd(a, s=None, axes=None)
inverse_real_fftnd(a, s=None, axes=None)
fft2d(a, s=None, axes=(-2,-1)) 
inverse_fft2d(a, s=None, axes=(-2, -1))
real_fft2d(a, s=None, axes=(-2,-1)) 
inverse_real_fft2d(a, s=None, axes=(-2, -1))
"""
